"""Host entry of the fused paired-CTA hd256 dK/dV kernel KV2 (fork kda-sm100-hd256kv; kernel dm_bwd_hd256kv2_sm100.py).
dm_bwd_hd256kv2(...) has the signature and outputs of bwd_hd256mx.dm_bwd_hd256mx: dQ from DmHd256Mx's dq mode (unchanged),
dK / dV from ONE DmHd256Kv2 launch (replacing the dv + dk modes).  Deterministic; no workspace, no atomics outside dbg.
"""
import torch

from . import bwd_hd256mx as _mx
from .dm_bwd_hd256kv2_sm100 import DmHd256Kv2

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


def kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None):
    B, N, H, Dq = q8.shape
    Hk = k8.shape[2]
    assert Dq == D and N % 128 == 0 and H % Hk == 0
    dev = q8.device
    causal = bool(causal)
    f8 = torch.float8_e4m3fn
    p = lambda x: x.view(f8).permute(1, 3, 2, 0)        # (B, N, h, D) -> (N, D, h, B)  K-major  # noqa: E731
    tt = lambda x: x.view(f8).permute(3, 1, 2, 0)       # (B, N, h, D) -> (D, N, h, B)  MN-major  # noqa: E731
    dk = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    dv = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    L = B * H
    big = dbg & 2
    z8 = lambda *s: torch.zeros(*s, device=dev, dtype=torch.uint8)  # noqa: E731
    dump = torch.zeros(L, N, N, 4, device=dev) if big else torch.zeros(1, 1, 1, 4, device=dev)
    cp_ = z8(L, N, N) if big else z8(1, 1, 16)
    cs_ = z8(L, N, N) if big else z8(1, 1, 16)
    sbp = torch.zeros(L, N, N // 32, device=dev, dtype=torch.int32) if big else torch.zeros(1, 1, 4, device=dev, dtype=torch.int32)
    sbs = torch.zeros_like(sbp)
    hh = torch.zeros(L, N, 4, device=dev, dtype=torch.int64) if dbg & 8 else torch.zeros(1, 1, 4, device=dev, dtype=torch.int64)
    ten = [p(k8), p(v8), p(q8), p(do8), tt(q8), tt(do8), qs, ks, vs, dos, lse, delta, dk, dv, dump, cp_, cs_, sbp, sbs, hh]
    lds = [1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2]
    kd = dbg & 10
    ot = DmHd256Kv2.DEFAULT_OPTS if opts is None else opts
    key = ("kv2", causal, kd, knob, ot)
    _mx._run(key, lambda: DmHd256Kv2(causal, kd, knob, ot), ten, lds, [N // 128, H, Hk])
    extra = dict(dump=dump, codesP=cp_, codesS=cs_, sbP=sbp, sbS=sbs, hash=hh)
    return dk, dv, extra


def dm_bwd_hd256kv2(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None):
    """Same contract as bwd_hd256mx.dm_bwd_hd256mx: (dq, dk, dv) [+ dbg dict].  dq = DmHd256Mx dq mode, dk/dv = KV2."""
    r = _mx.dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, modes=("dq",))
    dk, dv, ex = kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, knob=knob, opts=opts)
    if dbg:
        dq, _, _, d = r
        d["kv2"] = ex
        return dq, dk, dv, d
    return r[0], dk, dv
