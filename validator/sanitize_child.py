"""Run by validator/sanitize.py under compute-sanitizer: small shapes that stress tails and partial tiles, forward +
backward in both modes plus the debug hook.  Never imports the reference.  Shares the GPU with the gpupool filler
(sanitized runs are CPU-bound and untimed)."""
try:                    # gpupool: hand the GPU to the filler before this process even creates a CUDA context
    from gpupool_hook import gpupool_done as _gp_done
    _gp_done()
except ImportError:
    pass

import importlib.util
import os
import sys

import torch

torch.zeros(1, device="cuda")      # create the CUDA context first: the pool's auto-stop fires on context creation
try:
    from gpupool_hook import gpupool_done
    gpupool_done()                 # then hand the GPU back to the filler for the whole (CPU-bound, untimed) sanitized run
except ImportError:
    pass

spec = sys.argv[1]
target, _, func = spec.partition(":")
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
ms = importlib.util.spec_from_file_location("dm_candidate", target)
mod = importlib.util.module_from_spec(ms)
sys.modules["dm_candidate"] = mod
ms.loader.exec_module(mod)
fn = getattr(mod, func or "attn")
sup = getattr(mod, "SUPPORTS", {}) or {}
m = int(sup.get("seqlen_multiple", 128))
up = lambda n: max(m, (n + m - 1) // m * m)  # noqa: E731
SHAPES = [(1, 4, 4, up(1152)), (1, 4, 1, up(384)), (1, 8, 2, up(1408)), (2, 2, 2, up(128)), (1, 4, 4, up(1024))]
import time as _t
_sel = os.environ.get("DMVAL_SAN_SHAPE")                 # shard: run one shape only (racecheck shards fit the 20-min lease cap)
_t0 = _t.time()
for _i, (B, H, Hk, N) in enumerate(SHAPES):
    if _sel is not None and str(_i) != _sel:
        continue
    print(f"shape {_i} start B{B} H{H}/{Hk} N{N} t={_t.time() - _t0:.0f}s", flush=True)
    _cs = os.environ.get("DMVAL_SAN_CAUSAL")              # finer shard: "1" causal only, "0" non-causal only
    for causal in ((True, False) if sup.get("noncausal", True) else (True,)):
        if _cs is not None and str(int(causal)) != _cs:
            continue
        g = torch.Generator(device="cuda").manual_seed(N + H)
        q, k, v, do = [torch.randn(B, N, h, 128, device="cuda", generator=g).bfloat16() for h in (H, Hk, Hk, H)]
        modes = ["matched", "stale"] if causal or sup.get("matched_noncausal") else ["stale"]
        for mode in modes:
            l = [x.clone().requires_grad_(True) for x in (q, k, v)]
            o = fn(*l, causal=causal, mode=mode)
            o = o[0] if isinstance(o, (tuple, list)) else o
            o.backward(do)
            if mode == "matched" and hasattr(mod, "debug"):
                mod.debug(q, k, v, do, causal, mode)
            torch.cuda.synchronize()
        print("ok", B, H, Hk, N, causal, f"t={_t.time() - _t0:.0f}s", flush=True)
print("SANITIZE_CHILD_DONE", flush=True)
