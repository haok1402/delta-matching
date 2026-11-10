"""Host entry of the hd256-dqr backward (fork hd256-dqr; kda/hd256/dqr-draft.md): KV2 (DmHd256Kv2 + opt dqw) computes dK / dV and
stores its dS codes + raw block exponents to a workspace; DmHd256Dqg then forms dQ = dS . K as a pure block-scaled GEMM (no S / dP /
exp recompute).  Same contract and values as bwd_hd256kv2.dm_bwd_hd256kv2 (dq / dk / dv bitwise).  Opt-in: DM_H256_BWD_IMPL=dqr.
Workspace = 16 KiB per visited (q tile, key tile, q head) + 4 B, bounded by DM_H256_DQW_GB (GiB, default 1): the call runs in balanced
contiguous batch chunks (KV2 + Dqg per chunk, one reused workspace); a single batch element above the budget -> the bundle-6 path
(dm_bwd_hd256kv2).  Deterministic: one TMEM accumulation chain per dQ element, no atomics."""
import torch

from . import bwd_hd256mx as _mx
from . import bwd_hd256kv2 as _kv
from .dm_bwd_hd256dqg_sm100 import DmHd256Dqg

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


def _cfg():
    from .fastpath import env as _env
    return float(_env("DM_H256_DQW_GB", "1")), int(_env("DM_H256_DQG_STAGES", "3")), _env("DM_H256_DQW_MODE", "dqt")


def dqg_launch(W, Wsb, k8, ks, B, N, H, causal, *, knob="", stages=None, wsw=None, out=None):
    Hk = k8.shape[2]
    st = _cfg()[1] if stages is None else int(stages)
    f8 = torch.float8_e4m3fn
    dq = torch.empty(B, N, H, D, device=k8.device, dtype=torch.bfloat16) if out is None else out
    assert dq.shape == (B, N, H, D) and dq.is_contiguous()
    mA = W.view(f8).permute(2, 1, 0)                       # (T, key, q) -> (q, key, T): MN-major A (queries contiguous)
    mKt = k8.view(f8).permute(3, 1, 2, 0)                  # (B, N, Hk, 256) -> (256, N, Hk, B): MN-major B (as yr / tt)
    ten = [mA, mKt, Wsb, ks, dq]
    lds = [0, 0, 2, 2, 3]
    wsw = _cfg()[2] == "dqt" if wsw is None else bool(wsw)
    key = ("dqg", bool(causal), st, knob, wsw)
    _mx._run(key, lambda: DmHd256Dqg(bool(causal), st, knob, wsw), ten, lds, [N // 128, H, Hk], mode="dqg")
    return dq


def dqw_fits(B, H, N, causal):
    """True if at least one batch element's workspace fits the budget (the call is then split into balanced batch chunks)."""
    return _kv.dqw_tiles(1, H, N // 128, bool(causal)) * (16384 + 4) <= _cfg()[0] * (1 << 30)


def batch_chunks(B, H, N, causal):
    """balanced contiguous batch chunks whose workspace fits DM_H256_DQW_GB (GiB); same values per element (each (b, kv head)
    group's reductions stay inside one KV2 / Dqg launch), fixed order."""
    per_b = _kv.dqw_tiles(1, H, N // 128, bool(causal)) * (16384 + 4)
    bpc = max(1, min(B, int(_cfg()[0] * (1 << 30) // per_b)))
    nch = -(-B // bpc)
    sz = [B // nch + (1 if i < B % nch else 0) for i in range(nch)]
    out, b0 = [], 0
    for z in sz:
        out.append((b0, b0 + z))
        b0 += z
    return out


def dm_bwd_hd256dqr(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None, dqg_knob=""):
    """(dq, dk, dv) [+ dbg dict as dm_bwd_hd256kv2].  dbg != 0 or a batch element above the budget -> the bundle-6 path.
    Batch chunks (DM_H256_DQW_GB, default 1 GiB): for each chunk in ascending b: KV2(+dS workspace) then Dqg, one reused workspace."""
    B, N, H, _ = q8.shape
    Hk = k8.shape[2]
    causal = bool(causal)
    if dbg or not dqw_fits(B, H, N, causal):
        return _kv.dm_bwd_hd256kv2(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, knob=knob, opts=opts)
    mode = _cfg()[2]
    ot = (_kv.DmHd256Kv2.DEFAULT_OPTS if opts is None else opts) + " " + mode
    ch = batch_chunks(B, H, N, causal)
    bmax = max(b1 - b0 for b0, b1 in ch)
    W, Wsb = _kv.dqw_alloc(bmax, H, N, causal, q8.device)
    dq = torch.empty(B, N, H, D, device=q8.device, dtype=torch.bfloat16)
    dk = torch.empty(B, N, Hk, D, device=q8.device, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    for b0, b1 in ch:
        T = _kv.dqw_tiles(b1 - b0, H, N // 128, causal)
        ws = (W[:T], Wsb[:T])
        sl = lambda x: x[b0:b1]  # noqa: E731   (contiguous: batch is the outermost dim of every tensor)
        _kv.kv2_launch(*[sl(x) for x in (q8, qs, k8, ks, v8, vs, do8, dos, lse, delta)], causal, knob=knob, opts=ot, dqw=ws,
                       out=(sl(dk), sl(dv)))
        dqg_launch(*ws, sl(k8), sl(ks), b1 - b0, N, H, causal, knob=dqg_knob, wsw=mode == "dqt", out=sl(dq))
    return dq, dk, dv
