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

import torch

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


def fwd_bwd(fn, q, k, v, do, causal, mode):
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


def main():
    spec, job_path, out_path = sys.argv[1:4]
    job = torch.load(job_path, weights_only=False)
    global FWD_ONLY
    FWD_ONLY = bool(job.get("fwd_only", False))
    dev = torch.device(job.get("device", "cuda"))
    torch.cuda.set_device(dev)
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
    try:                                    # gpupool late stop: make sure the filler is gone before the candidate runs
        from gpupool_hook import gpupool_ready
        gpupool_ready()
    except ImportError:
        pass
    for cf in case_files:
        case = torch.load(cf, weights_only=False)
        r = run_case(mod, fn, case, dev)
        torch.save(r, cf + ".out")
        os.remove(cf)
        torch.cuda.empty_cache()
    leaked = [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    res["meta"]["leaked_modules"] = sorted(set(res["meta"]["leaked_modules"]) | set(leaked))
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
                    out = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                    nonfin = mism = 0
                    for _ in range(n - 1):
                        o2 = fwd_bwd(fn, q, k, v, do, case["causal"], mode)
                        nonfin += any(t is not None and not torch.isfinite(t).all() for t in o2)
                        mism += any(t is not None and not torch.equal(t, t0) for t, t0 in zip(o2, out))
                        del o2
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
                r["runs"]["%s/%s" % run] = [None if t is None else t.cpu() for t in out]
            dbg = getattr(mod, "debug", None)
            if dbg is not None and ("matched", "plain") in [tuple(x) for x in case["runs"]]:
                d = dbg(q, k, v, do, case["causal"], "matched")
                r["debug"] = {kk: vv.detach().cpu() for kk, vv in (d or {}).items() if torch.is_tensor(vv)}
        except Exception:
            r["error"] = traceback.format_exc()[-4000:]
        return r


if __name__ == "__main__":
    main()
