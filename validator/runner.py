"""Candidate side of the validator: runs in its own process, imports ONLY the candidate (never the reference), runs the
cases the parent wrote to a file, and writes the candidate's outputs back.  The parent (validate.py) never imports the
candidate, and the candidate never sees the reference or which outputs are expected.

  python runner.py <candidate spec> <job.pt> <out.pt>
  candidate spec: /path/file.py[:func] (default func `attn`) or pkg.module[:func]
"""
from __future__ import annotations
try:                    # optional GPU-sharing hook: hand the GPU to the filler before this process even creates a CUDA context
    from gpu_share_hook import share_done as _gp_done
    _gp_done()
except ImportError:
    pass


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
        import gpu_share_hook
        getattr(gpu_share_hook, name)()
    except ImportError:
        pass


def main():
    _hook("share_done")                   # startup (imports, candidate import, kernel-cache loads) is CPU work
    spec, job_path, out_path = sys.argv[1:4]
    job = torch.load(job_path, weights_only=False)
    global FWD_ONLY
    FWD_ONLY = bool(job.get("fwd_only", False))
    dev = torch.device(job.get("device", "cuda"))
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    _hook("share_done")                   # after context creation (auto-stop fires on it): share the GPU with the filler
    res = {"meta": {}, "cases": []}
    try:
        mod, fn = load(spec)
    except Exception:
        res["fatal"] = "import failed:\n" + traceback.format_exc()
        torch.save(res, out_path)
        return
    res["meta"]["NUMERICS"] = getattr(mod, "NUMERICS", None)
    res["meta"]["SUPPORTS"] = getattr(mod, "SUPPORTS", None)
    # declared determinism; a wrapper that only sets DET = False  is non-deterministic too
    res["meta"]["DETERMINISTIC"] = getattr(mod, "DETERMINISTIC", getattr(mod, "DET", True)) is not False
    leaked = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    res["meta"]["leaked_modules"] = leaked
    case_files = job.get("case_files", [])
    def hook(name):
        _trace(name)
        try:
            import gpu_share_hook
            getattr(gpu_share_hook, name)()
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
    # no gpu_share_hook ready call: validation is not timed, so the candidate shares the GPU with a co-scheduled filler job for the whole
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
    hook("share_done")
    for f in saves:
        f.result()
    io.shutdown()
    leaked = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    res["meta"]["leaked_modules"] = sorted(set(res["meta"]["leaked_modules"]) | set(leaked))
    _hook("share_done")
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
                    # repeated calls on the same inputs (races): no bitwise requirement; record per call the
                    # whole-tensor, per-128-row-block and per-element deviation from the first call, and keep every
                    # call's outputs for small cases so the parent can judge accuracy on the WORST call
                    n = int(kind.split(":")[1])
                    # concurrent load on the same GPU perturbs timing: a past nondet forward race (O off by 3e-2)
                    # showed up only this way (2/1200 calls, 0/2400 unloaded).  DMVAL_STRESS_LOAD=0 turns it off.
                    load = None
                    if os.environ.get("DMVAL_STRESS_LOAD", "1") != "0":
                        import subprocess as _sp
                        load = _sp.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "gpuload.py")], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                        import atexit
                        atexit.register(load.kill)          # never leave the load running if the runner dies
                    POISON_MODE[0] = "none"
                    out = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                    # mixed stress: alternate the primary inputs with an alt family; each call is
                    # compared with the first call of its own input set
                    alt = tuple(case[n_].to(dev) for n_ in ("aq", "ak", "av", "ado")) if "aq" in case else None
                    out_alt = fwd_bwd(fn, *alt, case["causal"], mode) if alt else None
                    # judge nd calls against the DET build's output for the same inputs (both modes are
                    # built); only a candidate without a det switch is compared with its own call 0 (validator: block 2e-3)
                    vs_det = False
                    if getattr(mod, "DETERMINISTIC", getattr(mod, "DET", True)) is False and hasattr(mod, "DET"):
                        _d0 = mod.DET
                        mod.DET = True
                        try:
                            ref_ = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                            ref_alt = fwd_bwd(fn, *alt, case["causal"], mode) if alt else None
                            vs_det = True
                        finally:
                            mod.DET = _d0
                    if vs_det:
                        out, out_alt = ref_, ref_alt
                    keep = q.shape[0] * q.shape[2] * q.shape[1] ** 2 <= 2 ** 28 and alt is None
                    calls = [[None if t is None else t.cpu() for t in out]] if keep else None
                    nonfin, gmax, bmax, emax = 0, 0.0, 0.0, 0.0
                    where = {}       # location of the worst block / element deviation (tensor, call, poison, block, head)
                    for it in range(n - 1):
                        POISON_MODE[0] = ("ff", "zero", "rand", "none")[it % 4]
                        use_alt = alt is not None and it % 2 == 1
                        o2 = fwd_bwd(fn, *(alt if use_alt else (q, k, v, do)), case["causal"], mode)
                        nonfin += any(t is not None and not torch.isfinite(t).all() for t in o2)
                        for tn, t, t0 in zip(("O", "dQ", "dK", "dV"), o2, out_alt if use_alt else out):
                            if t is None or t0 is None:
                                continue
                            a_, b_ = t.double(), t0.double()
                            gmax = max(gmax, ((a_ - b_).norm() / b_.norm().clamp_min(1e-300)).item())
                            B_, N_, H_, D_ = b_.shape
                            nb = (N_ + 127) // 128
                            pad = nb * 128 - N_
                            if pad:
                                a_ = torch.nn.functional.pad(a_, (0, 0, 0, 0, 0, pad))
                                b_ = torch.nn.functional.pad(b_, (0, 0, 0, 0, 0, pad))
                            da = (a_ - b_).reshape(B_, nb, 128, H_, D_).pow(2).sum((2, 4)).sqrt()
                            db = b_.reshape(B_, nb, 128, H_, D_).pow(2).sum((2, 4)).sqrt()
                            # normalize by max(block norm, head-wide RMS of the block norms), so blocks
                            # whose values nearly cancel (growmax) do not look like races; a lost tile is still ~100x above
                            rms = db.pow(2).mean(dim=1, keepdim=True).sqrt()           # (B, 1, H)
                            den_ = torch.maximum(db, rms)
                            br = torch.where(den_ > 0, da / den_.clamp_min(1e-30), torch.zeros_like(da))
                            bv = br.max().item()
                            if bv > bmax:
                                ix = int(br.argmax())
                                where["block"] = {"tensor": tn, "call": it + 1, "poison": POISON_MODE[0], "rel": bv,
                                                  "inputs": "alt" if use_alt else "primary",
                                                  "batch": ix // (nb * H_), "block": (ix // H_) % nb, "head": ix % H_}
                            bmax = max(bmax, bv)
                            rowmax = b_.abs().amax(-1, keepdim=True).clamp_min(1e-30)
                            ev = ((a_ - b_).abs() / rowmax).max().item()
                            if ev > emax:
                                where["elem"] = {"tensor": tn, "call": it + 1, "poison": POISON_MODE[0], "rel": ev}
                            emax = max(emax, ev)
                        if keep:
                            calls.append([None if t is None else t.cpu() for t in o2])
                        del o2
                    POISON_MODE[0] = "none"
                    if load is not None:
                        load.kill()
                        load.wait()
                    nonfin += any(t is not None and not torch.isfinite(t).all() for t in out)
                    r.setdefault("stress", {})["%s/%s" % run] = {"calls": n, "nonfinite_calls": nonfin, "mismatch_calls": 0,
                                                                 "rel_max": gmax, "block_rel_max": bmax, "elem_rel_max": emax,
                                                                 "vs": "det" if vs_det else "call0",
                                                                 "where": where}
                    if keep:
                        r.setdefault("stress_calls", {})["%s/%s" % run] = calls
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
                d2 = dbg(q, k, v, do, case["causal"], "matched")      # repeat: identity must hold on EVERY run
                r["debug_repeat"] = {kk: vv.detach().cpu() for kk, vv in (d2 or {}).items() if torch.is_tensor(vv)}
        except Exception:
            r["error"] = traceback.format_exc()[-4000:]
        return r


if __name__ == "__main__":
    main()

