"""Host entry of the hd256-dqr backward (fork hd256-dqr; kda/hd256/dqr-draft.md): KV2 (DmHd256Kv2 + opt dqw) computes dK / dV and
stores its dS codes + raw block exponents to a workspace; DmHd256Dqg then forms dQ = dS . K as a pure block-scaled GEMM (no S / dP /
exp recompute).  Same contract and values as bwd_hd256kv2.dm_bwd_hd256kv2 (dq / dk / dv bitwise).  Opt-in: DM_H256_BWD_IMPL=dqr.
Workspace = 16 KiB per visited (q tile, key tile, q head) + 4 B: used only when <= DM_H256_DQW_GB (default 9) GiB, else this call
falls back to the bundle-6 path (dm_bwd_hd256kv2).  Deterministic: one TMEM accumulation chain per dQ element, no atomics."""
import torch

from . import bwd_hd256mx as _mx
from . import bwd_hd256kv2 as _kv
from .dm_bwd_hd256dqg_sm100 import DmHd256Dqg

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


def _cfg():
    from .fastpath import env as _env
    return float(_env("DM_H256_DQW_GB", "9")), int(_env("DM_H256_DQG_STAGES", "3"))


def dqg_launch(W, Wsb, k8, ks, B, N, H, causal, *, knob="", stages=None):
    Hk = k8.shape[2]
    st = _cfg()[1] if stages is None else int(stages)
    f8 = torch.float8_e4m3fn
    dq = torch.empty(B, N, H, D, device=k8.device, dtype=torch.bfloat16)
    mA = W.view(f8).permute(2, 1, 0)                       # (T, key, q) -> (q, key, T): MN-major A (queries contiguous)
    mKt = k8.view(f8).permute(3, 1, 2, 0)                  # (B, N, Hk, 256) -> (256, N, Hk, B): MN-major B (as yr / tt)
    ten = [mA, mKt, Wsb, ks, dq]
    lds = [0, 0, 2, 2, 3]
    key = ("dqg", bool(causal), st, knob)
    _mx._run(key, lambda: DmHd256Dqg(bool(causal), st, knob), ten, lds, [N // 128, H, Hk], mode="dqg")
    return dq


def dqw_fits(B, H, N, causal):
    T = _kv.dqw_tiles(B, H, N // 128, bool(causal))
    return T * (16384 + 4) <= _cfg()[0] * (1 << 30)


def dm_bwd_hd256dqr(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None, dqg_knob=""):
    """(dq, dk, dv) [+ dbg dict as dm_bwd_hd256kv2].  dbg != 0 -> the bundle-6 path (dq kernel dumps / identity / lockstep live there)."""
    B, N, H, _ = q8.shape
    causal = bool(causal)
    if dbg or not dqw_fits(B, H, N, causal):
        return _kv.dm_bwd_hd256kv2(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, knob=knob, opts=opts)
    ot = (_kv.DmHd256Kv2.DEFAULT_OPTS if opts is None else opts) + " dqw"
    W, Wsb = _kv.dqw_alloc(B, H, N, causal, q8.device)
    dk, dv, _ = _kv.kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, knob=knob, opts=ot, dqw=(W, Wsb))
    dq = dqg_launch(W, Wsb, k8, ks, B, N, H, causal, knob=dqg_knob)
    return dq, dk, dv
