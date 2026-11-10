"""Host entry of the hd256-dqr backward (fork hd256-dqr; kda/hd256/dqr-draft.md): KV2 (DmHd256Kv2 + opt dqw) computes dK / dV and
stores its dS codes + raw block exponents to a workspace; DmHd256Dqg then forms dQ = dS . K as a pure block-scaled GEMM (no S / dP /
exp recompute).  Same contract and values as bwd_hd256kv2.dm_bwd_hd256kv2 (dq / dk / dv bitwise).  The api DEFAULT since
hd256-dqr2 (DM_H256_BWD_IMPL=kv2 restores the bundle-6/7 path everywhere).  Workspace = 16 KiB per visited (q tile, key tile, q head)
+ 4 B, bounded by DM_H256_DQW_GB (GiB, default 2 = the kda-hd256 scratch cap, never exceeded): the call runs in balanced contiguous batch chunks (KV2 + Dqg per chunk, one reused workspace); a single batch element above the budget -> the bundle-6 path
(dm_bwd_hd256kv2).  Deterministic: one TMEM accumulation chain per dQ element, no atomics."""
import torch

from . import bwd_hd256mx as _mx
from . import bwd_hd256kv2 as _kv
from .dm_bwd_hd256dqg_sm100 import DmHd256Dqg

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


def _cfg():
    """(scratch cap GiB, Dqg stages, KV2 store mode).  Cap: DM_BWD_WS_LIMIT_GB (the sm90 split.py name; provisional hd256 default 2 GiB),
    DM_H256_DQW_GB = the older hd256-dqr name, honoured if DM_BWD_WS_LIMIT_GB is unset."""
    from .fastpath import env as _env
    cap = _env("DM_BWD_WS_LIMIT_GB", None)
    cap = _env("DM_H256_DQW_GB", "2") if cap is None else cap
    return float(cap), int(_env("DM_H256_DQG_STAGES", "3")), _env("DM_H256_DQW_MODE", "dqt")


def _groups_mode():
    """DM_BWD_HEAD_GROUPS (as sm90 split.py): "auto" (default) = whole call, else batch chunks, else KV-head-group chunks, whichever
    is the first to fit the cap; "off" = never split heads (batch chunks only); <n> = force n KV-head groups per call."""
    from .fastpath import env as _env
    v = str(_env("DM_BWD_HEAD_GROUPS", "auto")).strip().lower()
    assert v in ("auto", "off") or v.isdigit(), v
    return v


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


def _ws_bytes(B, h, N, causal):
    return _kv.dqw_tiles(B, h, N // 128, bool(causal)) * (16384 + 4)


def _tmp_bytes(B, N, r, ng):
    """head-split chunk temporaries (contiguous copies of the chunk's heads): q8 + do8 + dq (bf16) per q head, k8 + v8 + dk + dv
    (bf16) per kv head, + scales / lse / delta; counted against the cap together with the workspace."""
    return B * N * (ng * r * 256 * 4 + ng * 256 * 6 + ng * r * 16 + ng * 16)


def _balanced(n, per):
    nch = -(-n // max(1, min(n, per)))
    sz = [n // nch + (1 if i < n % nch else 0) for i in range(nch)]
    out, a = [], 0
    for z in sz:
        out.append((a, a + z))
        a += z
    return out


def plan(B, H, Hk, N, causal):
    """Chunk plan [(b0, b1, g0, g1), ...] (batch range x KV-head-group range, ascending b then g) whose workspace (+ head-split
    temporaries) fits the cap, or None (-> the kv2 fallback).  Every (b, KV group)'s reductions (dK / dV over its q heads, dQ over keys)
    stay inside one KV2 / Dqg launch, so any plan gives the unsplit values bitwise."""
    cap = _cfg()[0] * (1 << 30)
    r = H // Hk
    gm = _groups_mode()
    if gm.isdigit():                                   # forced n groups per call
        ng = max(1, min(Hk, int(gm)))
        if ng == Hk:
            gm = "off"
        else:
            if _ws_bytes(1, ng * r, N, causal) + _tmp_bytes(1, N, r, ng) > cap:
                return None
            return [(b, b + 1, g0, g1) for b in range(B) for g0, g1 in _balanced(Hk, ng)]
    per_b = _ws_bytes(1, H, N, causal)
    if per_b <= cap:                                   # whole call or balanced batch chunks (no copies)
        return [(b0, b1, 0, Hk) for b0, b1 in _balanced(B, int(cap // per_b))]
    if gm == "off":
        return None
    ng = 0
    for n in range(Hk - 1, 0, -1):                     # largest KV-group count per chunk that fits
        if _ws_bytes(1, n * r, N, causal) + _tmp_bytes(1, N, r, n) <= cap:
            ng = n
            break
    if ng == 0:
        return None
    return [(b, b + 1, g0, g1) for b in range(B) for g0, g1 in _balanced(Hk, ng)]


def dqw_fits(B, H, N, causal, Hk=None):
    """True if the A2 path runs (a chunk plan fits the cap).  Hk None: batch chunks only (older callers)."""
    if Hk is None:
        return _ws_bytes(1, H, N, causal) <= _cfg()[0] * (1 << 30)
    return plan(B, H, Hk, N, causal) is not None


def batch_chunks(B, H, N, causal):
    """(older callers) balanced contiguous batch chunks whose workspace fits the cap."""
    return _balanced(B, int(_cfg()[0] * (1 << 30) // _ws_bytes(1, H, N, causal)))


def dm_bwd_hd256dqr(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None, dqg_knob=""):
    """(dq, dk, dv) [+ dbg dict as dm_bwd_hd256kv2].  dbg != 0 or no chunk plan within the cap -> the kv2 (bundle-6/7/8) path.
    Chunks (plan(): batch ranges, or (b, KV-group range) above the per-batch-element cap), ascending: KV2(+dS workspace) then Dqg per
    chunk, one reused workspace (stream order = the W hand-off between chunks)."""
    B, N, H, _ = q8.shape
    Hk = k8.shape[2]
    r = H // Hk
    causal = bool(causal)
    ch = None if dbg else plan(B, H, Hk, N, causal)
    if ch is None:
        return _kv.dm_bwd_hd256kv2(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, knob=knob, opts=opts)
    mode = _cfg()[2]
    ot = (_kv.DmHd256Kv2.DEFAULT_OPTS if opts is None else opts) + " " + mode
    Tmax = max(_kv.dqw_tiles(b1 - b0, (g1 - g0) * r, N // 128, causal) for b0, b1, g0, g1 in ch)
    W = torch.empty(Tmax, 128, 128, device=q8.device, dtype=torch.uint8)
    Wsb = torch.empty(Tmax, 1, 4, device=q8.device, dtype=torch.uint8)
    dq = torch.empty(B, N, H, D, device=q8.device, dtype=torch.bfloat16)
    dk = torch.empty(B, N, Hk, D, device=q8.device, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    for b0, b1, g0, g1 in ch:
        nb, ng = b1 - b0, g1 - g0
        T = _kv.dqw_tiles(nb, ng * r, N // 128, causal)
        ws = (W[:T], Wsb[:T])
        if ng == Hk:     # batch chunk: contiguous batch slices, no copies
            sl = lambda x: x[b0:b1]  # noqa: E731
            _kv.kv2_launch(*[sl(x) for x in (q8, qs, k8, ks, v8, vs, do8, dos, lse, delta)], causal, knob=knob, opts=ot, dqw=ws,
                           out=(sl(dk), sl(dv)))
            dqg_launch(*ws, sl(k8), sl(ks), nb, N, H, causal, knob=dqg_knob, wsw=mode == "dqt", out=sl(dq))
            continue
        hq, hk = slice(g0 * r, g1 * r), slice(g0, g1)   # KV-head-group chunk: contiguous copies of the chunk's heads
        tq = lambda x: x[b0:b1, :, hq].contiguous()  # noqa: E731   (B, N, h, D)
        tk = lambda x: x[b0:b1, :, hk].contiguous()  # noqa: E731
        sq = lambda x: x[b0:b1, hq].contiguous()  # noqa: E731     (B, h, ...)
        sk = lambda x: x[b0:b1, hk].contiguous()  # noqa: E731
        k8c, ksc = tk(k8), sk(ks)
        dkc, dvc, _ = _kv.kv2_launch(tq(q8), sq(qs), k8c, ksc, tk(v8), sk(vs), tq(do8), sq(dos), sq(lse), sq(delta), causal, knob=knob,
                                     opts=ot, dqw=ws)
        dqc = dqg_launch(*ws, k8c, ksc, nb, N, ng * r, causal, knob=dqg_knob, wsw=mode == "dqt")
        dq[b0:b1, :, hq].copy_(dqc)
        dk[b0:b1, :, hk].copy_(dkc)
        dv[b0:b1, :, hk].copy_(dvc)
        del dqc, dkc, dvc, k8c, ksc
    return dq, dk, dv
