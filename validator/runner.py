"""Candidate side of the validator: runs in its own process, imports ONLY the candidate (never the reference), runs the
cases the parent wrote to a file, and writes the candidate's outputs back.  The parent (validate.py) never imports the
candidate, and the candidate never sees the reference or which outputs are expected.

  python runner.py <candidate spec> <job.pt> <out.pt>
  candidate spec: /path/file.py[:func] (default func `attn`) or pkg.module[:func]
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import traceback

import time as _time
import torch


def _trace(msg):
    path = os.environ.get("DMVAL_TRACE")       # a file path: append "HH:MM:SS.mmm <file> <event>" lines
    if path:
        with open(path, "a") as f:
            f.write(f"{_time.strftime('%H:%M:%S')}.{int(_time.time() * 1000) % 1000:03d} {os.path.basename(__file__)} {msg}\n")

FORBIDDEN = ("delta_matching.reference", "validator", "reference")


def load(spec):
    target, _, func = spec.partition(":")
    func = func or "attn"
    if target.endswith(".py"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
        ms = importlib.util.spec_from_file_location("dm_candidate", target)
        mod = importlib.util.module_from_spec(ms)
        sys.modules["dm_candidate"] = mod
        ms.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return mod, getattr(mod, func)


def call(fn, q, k, v, causal, mode):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {"mode": None, "softmax_scale": None}
    kw = {"causal": causal}
    if "mode" in params:
        kw["mode"] = mode
    if "softmax_scale" in params:
        kw["softmax_scale"] = None
    o = fn(q, k, v, **kw)
    return o[0] if isinstance(o, (tuple, list)) else o


FWD_ONLY = False
POISON_BYTES = int(os.environ.get("DMVAL_POISON_GB", "8")) << 30


POISON_MODE = ["none"]
BIG_CASE = 2 ** 31            # B * H * N^2 above which a case takes the GPU exclusively


def poison():
    """Set the caching allocator's free memory (large and small pools) to a pattern before a candidate call, so that
    out-of-bounds reads see different garbage on different calls of a stress run: "none" (whatever is there), "ff" (0xFF:
    e4m3 / bf16 / fp32 NaN patterns), "zero", "rand".  One fixed pattern can HIDE an OOB bug (0xFF hid one), so stress
    runs cycle all of them and require every call finite and bitwise equal."""
    mode = POISON_MODE[0]
    if mode == "none":
        return
    free, _ = torch.cuda.mem_get_info()
    n = min(POISON_BYTES, max(0, free - (8 << 30)))
    bufs = [torch.empty(n, dtype=torch.uint8, device="cuda")] if n > 0 else []
    bufs += [torch.empty(1 << 20, dtype=torch.uint8, device="cuda") for _ in range(256)]   # small-pool segments
    for x in bufs:
        if mode == "ff":
            x.fill_(0xFF)
        elif mode == "zero":
            x.zero_()
        else:
            x.random_(0, 256)
    del bufs


def fwd_bwd(fn, q, k, v, do, causal, mode):
    poison()
    if FWD_ONLY:
        with torch.no_grad():
            o = call(fn, q.detach().clone(), k.detach().clone(), v.detach().clone(), causal, mode)
        torch.cuda.synchronize()
        return [o.detach().clone(), None, None, None]
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    o = call(fn, *leaves, causal, mode)
    o.backward(do)
    torch.cuda.synchronize()
    return [o.detach().clone()] + [x.grad.detach().clone() if x.grad is not None else None for x in leaves]


def fwd_bwd_views(fn, q, k, v, do, causal, mode):
    """q / k / v as strided views of one packed (B, N, H + 2Hk, D) buffer (flash-attention's usual call)."""
    H, Hk = q.shape[2], k.shape[2]
    poison()
    if FWD_ONLY:
        qkv = torch.cat([q, k, v], dim=2)
        with torch.no_grad():
            o = call(fn, *qkv.split([H, Hk, Hk], dim=2), causal, mode)
        torch.cuda.synchronize()
        return [o.detach().clone(), None, None, None]
    qkv = torch.cat([q, k, v], dim=2).detach().requires_grad_(True)
    qv, kv, vv = qkv.split([H, Hk, Hk], dim=2)
    o = call(fn, qv, kv, vv, causal, mode)
    o.backward(do)
    torch.cuda.synchronize()
    g = qkv.grad
    return [o.detach().clone(), g[:, :, :H].clone(), g[:, :, H:H + Hk].clone(), g[:, :, H + Hk:].clone()]


def _hook(name):
    try:
        import gpupool_hook
        getattr(gpupool_hook, name)()
    except ImportError:
        pass


def main():
    _hook("gpupool_done")                   # startup (imports, candidate import, kernel-cache loads) is CPU work
    spec, job_path, out_path = sys.argv[1:4]
    job = torch.load(job_path, weights_only=False)
    global FWD_ONLY
    FWD_ONLY = bool(job.get("fwd_only", False))
    dev = torch.device(job.get("device", "cuda"))
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    _hook("gpupool_done")                   # after context creation (auto-stop fires on it): share the GPU with the filler
    res = {"meta": {}, "cases": []}
    try:
        mod, fn = load(spec)
    except Exception:
        res["fatal"] = "import failed:\n" + traceback.format_exc()
        torch.save(res, out_path)
        return
    res["meta"]["NUMERICS"] = getattr(mod, "NUMERICS", None)
    res["meta"]["SUPPORTS"] = getattr(mod, "SUPPORTS", None)
    leaked = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    res["meta"]["leaked_modules"] = leaked
    case_files = job.get("case_files", [])
    def hook(name):
        _trace(name)
        try:
            import gpupool_hook
            getattr(gpupool_hook, name)()
        except ImportError:
            pass
    # one GPU phase for all cases: case files are read ahead and outputs written behind in threads (/dev/shm), so the
    # GPU never waits on file I/O and the filler is not toggled for sub-second CPU windows (it cannot restart that fast)
    from concurrent.futures import ThreadPoolExecutor
    io = ThreadPoolExecutor(max_workers=2)
    loads = [io.submit(torch.load, cf, weights_only=False) for cf in case_files[:2]]
    saves = []

    def save(r, cf):
        r["runs"] = {kk: [None if t is None else t for t in vv] for kk, vv in r["runs"].items()}
        torch.save(r, cf + ".out")
        os.remove(cf)
    # no gpupool_ready(): validation is not timed, so the candidate shares the GPU with the gpupool filler for the whole
    # run (the GPU is never idle while this process waits on I/O, kernel-cache loads or Python)
    for i, cf in enumerate(case_files):
        case = loads[i].result()
        if i + 2 < len(case_files):
            loads.append(io.submit(torch.load, case_files[i + 2], weights_only=False))
        _trace("case start " + case["name"])
        r = run_case(mod, fn, case, dev)
        # device-to-host copies here (fast, pinned-free but ~GB/s x10); the file write happens in the I/O thread
        r["runs"] = {kk: [None if t is None else t.cpu() for t in vv] for kk, vv in r["runs"].items()}
        _trace("case end " + case["name"])
        saves.append(io.submit(save, r, cf))
        del case
        torch.cuda.empty_cache()
    hook("gpupool_done")
    for f in saves:
        f.result()
    io.shutdown()
    leaked = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    res["meta"]["leaked_modules"] = sorted(set(res["meta"]["leaked_modules"]) | set(leaked))
    _hook("gpupool_done")
    torch.save(res, out_path)


def run_case(mod, fn, case, dev):
    if True:
        r = {"name": case["name"], "runs": {}}
        try:
            q, k, v, do = (case[n].to(dev) for n in ("q", "k", "v", "do"))
            for run in case["runs"]:          # e.g. ("matched", "plain"), ("matched", "repeat"), ("stale", "views")
                mode, kind = run
                if kind == "views":
                    out = fwd_bwd_views(fn, q, k, v, do, case["causal"], mode)
                elif kind.startswith("stress"):
                    # repeated calls on the same inputs: every call finite and bitwise equal to the first (races)
                    n = int(kind.split(":")[1])
                    POISON_MODE[0] = "none"
                    out = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                    nonfin = mism = 0
                    for it in range(n - 1):
                        POISON_MODE[0] = ("ff", "zero", "rand", "none")[it % 4]
                        o2 = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                        nonfin += any(t is not None and not torch.isfinite(t).all() for t in o2)
                        mism += any(t is not None and not torch.equal(t, t0) for t, t0 in zip(o2, out))
                        del o2
                    POISON_MODE[0] = "none"
                    nonfin += any(t is not None and not torch.isfinite(t).all() for t in out)
                    r.setdefault("stress", {})["%s/%s" % run] = {"calls": n, "nonfinite_calls": nonfin,
                                                                 "mismatch_calls": mism}
                elif kind == "mutate":
                    # caching probe: run on buffers, overwrite them in place with the case's second input set, run again
                    bufs = [x.clone() for x in (q, k, v)]
                    fwd_bwd(fn, *bufs, do, case["causal"], mode)
                    for b, n in zip(bufs, ("q2", "k2", "v2")):
                        b.copy_(case[n].to(dev))
                    out = fwd_bwd(fn, *bufs, do, case["causal"], mode)
                else:
                    out = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                r["runs"]["%s/%s" % run] = [None if t is None else t.detach() for t in out]     # host copy after done()
            dbg = getattr(mod, "debug", None)
            if dbg is not None and ("matched", "plain") in [tuple(x) for x in case["runs"]]:
                d = dbg(q, k, v, do, case["causal"], "matched")
                r["debug"] = {kk: vv.detach().cpu() for kk, vv in (d or {}).items() if torch.is_tensor(vv)}
        except Exception:
            r["error"] = traceback.format_exc()[-4000:]
        return r


if __name__ == "__main__":
    main()

