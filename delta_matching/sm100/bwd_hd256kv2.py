"""
Host entry of the fused paired-CTA hd256 dK / dV kernel KV2: kv2_launch runs
one DmHd256Kv2 launch, composed into the backward by attention_bwd_hd256_sm100.
Deterministic; no atomics outside the debug builds.
"""
import torch

from . import bwd_hd256mx as _mx
from .dm_bwd_hd256kv2_sm100 import DmHd256Kv2

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


_PH = {}


def _ph(dev):
    """per-device dbg placeholders (never written by a dbg=0 build)."""
    k = str(dev)
    if k not in _PH:
        _PH[k] = dict(dump=torch.zeros(1, 1, 1, 4, device=dev, dtype=torch.float32),
                      c=torch.zeros(1, 1, 16, device=dev, dtype=torch.uint8),
                      sb=torch.zeros(1, 1, 4, device=dev, dtype=torch.int32),
                      h=torch.zeros(1, 1, 4, device=dev, dtype=torch.int64))
    return _PH[k]


def dqw_tiles(B, H, n_t, causal, n_k=None):
    """number of (l, q tile, key tile) tiles of the dQ workspace (n_k: key tiles of a non-causal rect block)."""
    if n_k is not None and n_k != n_t:
        if causal:          # bottom-right aligned (ring chunk): q tile ct reads key tiles 0 .. ct + n_k - n_t
            assert n_k > n_t
            return B * H * (n_t * (n_t + 1) // 2 + n_t * (n_k - n_t))
        return B * H * n_t * n_k
    return B * H * (n_t * (n_t + 1) // 2 if causal else n_t * n_t)


def dqw_alloc(B, H, N, causal, dev):
    T = dqw_tiles(B, H, N // 128, bool(causal))
    return (torch.empty(T, 128, 128, device=dev, dtype=torch.uint8), torch.empty(T, 1, 4, device=dev, dtype=torch.uint8))


def kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None, dqw=None, out=None, dqc=None):
    """dqw = (W, Wsb) from dqw_alloc: KV2 also stores the dS codes / raw block exponents for the dQ GEMM (opts must contain dqw).
    dqc = (desc, carry_dk, carry_dv, q1): exact-carry chunk, see DmHd256Kv2 opt dqc; desc =
    int32 device tensor [Q0, n_t, cin, cout] (+ pad), carries fp32 (B, N, Hk, 256); the launch covers q tiles [Q0, q1)."""
    B, N, H, Dq = q8.shape
    Hk = k8.shape[2]
    Nk = k8.shape[1]
    assert Dq == D and N % 128 == 0 and H % Hk == 0 and Nk % 128 == 0 and v8.shape == k8.shape
    dev = q8.device
    causal = bool(causal)
    ot_ = DmHd256Kv2.DEFAULT_OPTS if opts is None else opts
    rect = "rect" in ot_.split()
    assert rect or Nk == N, "Nq != Nk needs opt rect (ring blocks)"
    assert not (rect and causal and Nk < N), "causal rect: bottom-right aligned, Nk >= Nq"
    assert qs.shape == (B, H, N // 32) and dos.shape == qs.shape and lse.shape == (B, H, N) and delta.shape == lse.shape
    assert ks.shape == (B, Hk, Nk // 32) and vs.shape == ks.shape
    f8 = torch.float8_e4m3fn
    p = lambda x: x.view(f8).permute(1, 3, 2, 0)        # (B, N, h, D) -> (N, D, h, B)  K-major  # noqa: E731
    tt = lambda x: x.view(f8).permute(3, 1, 2, 0)       # (B, N, h, D) -> (D, N, h, B)  MN-major  # noqa: E731
    if out is not None:      # (dk, dv) contiguous batch slices of the caller's outputs
        dk, dv = out
        assert dk.shape == (B, Nk, Hk, D) and dk.is_contiguous() and dv.shape == dk.shape and dv.is_contiguous()
        f32 = "dkf32" in ot_.split() or "dkacc" in ot_.split()
        assert dk.dtype == dv.dtype == (torch.float32 if f32 else torch.bfloat16), (dk.dtype, ot_)
    else:
        assert "dkacc" not in ot_.split(), "dkacc needs the caller's fp32 accumulators (out=)"
        odt = torch.float32 if "dkf32" in ot_.split() else torch.bfloat16
        dk = torch.empty(B, Nk, Hk, D, device=dev, dtype=odt)
        dv = torch.empty(B, Nk, Hk, D, device=dev, dtype=odt)
    L = B * H
    big = dbg & 2
    if big or dbg & 24:
        z8 = lambda *s_: torch.zeros(*s_, device=dev, dtype=torch.uint8)  # noqa: E731
        dump = torch.zeros(L, N, N, 4, device=dev, dtype=torch.float32) if big else _ph(dev)["dump"]
        cp_ = z8(L, N, N) if big else _ph(dev)["c"]
        cs_ = z8(L, N, N) if big else _ph(dev)["c"]
        sbp = torch.zeros(L, N, N // 32, device=dev, dtype=torch.int32) if big else _ph(dev)["sb"]
        sbs = torch.zeros_like(sbp) if big else _ph(dev)["sb"]
        hh = torch.zeros(L, N, 4, device=dev, dtype=torch.int64) if dbg & 24 else _ph(dev)["h"]   # 16: [rowsum dS, sum|P dP|] f64
    else:   # production: the dbg tensors are never touched (compiled out) -> shared placeholders, no per-call zeroing
        ph = _ph(dev)
        dump, cp_, cs_, sbp, sbs, hh = ph["dump"], ph["c"], ph["c"], ph["sb"], ph["sb"], ph["h"]
    ot = DmHd256Kv2.DEFAULT_OPTS if opts is None else opts
    assert (dqw is not None) == ("dqw" in ot.split() or "dqt" in ot.split()), (ot, dqw is None)
    if dqw is not None:
        assert not big
        cs_, sbs = dqw
        assert (dqc is not None or cs_.shape == (dqw_tiles(B, H, N // 128, causal, Nk // 128), 128, 128)) and sbs.shape == (cs_.shape[0], 1, 4)
    assert (dqc is not None) == ("dqc" in ot.split())
    nt_arg = N // 128
    if dqc is not None:
        desc, ck, cv, nt_arg = dqc
        assert desc.dtype == torch.int32 and desc.numel() >= 4 and ck.dtype == torch.float32 and ck.shape == (B, N, Hk, D)
        assert cv.shape == ck.shape and ck.is_contiguous() and cv.is_contiguous() and 1 <= nt_arg <= N // 128
        dump, cp_, sbp = ck, cv.view(torch.uint8).view(B * N, Hk, D * 4), desc.view(1, 1, -1)
    ten = [p(k8), p(v8), p(q8), p(do8), tt(q8), tt(do8), qs, ks, vs, dos, lse, delta, dk, dv, dump, cp_, cs_, sbp, sbs, hh]
    lds = [1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2]
    kd = dbg & 26
    key = ("kv2", causal, kd, knob, ot)
    _mx._run(key, lambda: DmHd256Kv2(causal, kd, knob, ot), ten, lds, [nt_arg, H, Hk])
    extra = dict(dump=dump, codesP=cp_, codesS=cs_, sbP=sbp, sbS=sbs, hash=hh)
    return dk, dv, extra

