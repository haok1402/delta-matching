"""Development reference of the sm100-mx numerics (kda-sm100's own; dm-validator's reference.py is authoritative).

Inputs are the producers' codes/scales (quant.py).  Everything in fp32 / fp64 torch, (B, H, N, D) internally.
"""
import math
import torch

LOG2E = 1.4426950408889634
BLK = 32


def _deq(codes, scales):
    # codes (B, N, H, D) e4m3, scales (B, H, N/32) -> (B, H, N, D) fp32
    B, N, H, D = codes.shape
    x = codes.float().permute(0, 2, 1, 3)
    return x * scales.repeat_interleave(BLK, dim=2)[..., None]


def _pow2_floor_exp(x):
    """floor(log2 x) for x > 0 fp32, exact via frexp."""
    m, e = torch.frexp(x)
    return (e - 1).to(torch.float32)


def fwd_ref(q8, qs, k8, ks, v8, vs, causal, emulate_p=True, l_from_codes=False):
    """O (B, N, H, D) fp32, LSE (B, H, N) natural, plus P-code fingerprint data.

    Scores x = (Q8 K8^T) * qs * ks * log2e (log2 units).  Per (row, 32-key block): kb = floor(max x);
    codes = rn_e4m3(exp2(x - kb + 7)); weight 2^(kb - 7).  O = sum codes*weight*V / l,
    l = sum exp2(x - kb + 7) (fp32 pre-rounding) * weight.
    """
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    rep = H // Hk
    Q = _deq(q8, qs)
    K = _deq(k8, ks).repeat_interleave(rep, dim=1)
    V = _deq(v8, vs).repeat_interleave(rep, dim=1)
    x = torch.matmul(Q, K.transpose(-1, -2)) * LOG2E            # (B, H, N, N) log2 units
    if causal:
        mask = torch.ones(N, N, dtype=torch.bool, device=x.device).tril()
        x = x.masked_fill(~mask, float("-inf"))
    xb = x.view(B, H, N, N // BLK, BLK)
    bmax = xb.amax(dim=-1, keepdim=True)
    kb = torch.where(torch.isfinite(bmax), torch.floor(bmax), torch.zeros_like(bmax))
    pt = torch.exp2(xb - kb + 7)                                  # [0, 256)
    w = torch.exp2(kb - 7)
    if emulate_p:
        pc = pt.to(torch.float8_e4m3fn).float()
    else:
        pc = pt
    # global max for numerically-safe normalization (exact pow2 shifts)
    m = torch.amax(kb, dim=-2, keepdim=True)
    w = torch.exp2(kb - 7 - m)
    l = ((pc if l_from_codes else pt).sum(-1, keepdim=True) * w).sum(-2).squeeze(-1)   # (B, H, N); lq/cp: code sum
    P = (pc * w).view(B, H, N, N)
    O = torch.matmul(P, V) / l[..., None]
    lse = (m.squeeze(-1).squeeze(-1) + torch.log2(l)) / LOG2E
    return O.permute(0, 2, 1, 3), lse


def _pow2_ceil_scale(amax):
    """Smallest power of 2 s with amax <= 448 s (fp32 bits: amax = m 2^E -> 2^(E-8) if m <= 1.75 else 2^(E-7)); 1 if amax == 0."""
    ab = amax.float().contiguous().view(torch.int32)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(torch.int32)
    eb = eb.clamp(min=1, max=254)
    eb = torch.where(amax > 0, eb, torch.full_like(eb, 127))
    return (eb << 23).view(torch.float32)


def _q_blocks(x, bm, bn):
    """e4m3 quantize-dequantize x (..., M, N) with one pow2 scale per (bm x bn) block."""
    *lead, M, N = x.shape
    xb = x.reshape(*lead, M // bm, bm, N // bn, bn)
    s = _pow2_ceil_scale(xb.abs().amax(dim=(-3, -1), keepdim=True))
    return ((xb / s).to(torch.float8_e4m3fn).float() * s).reshape(*lead, M, N)


def bwd_ref(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, sm):
    """sm100-mx backward (dev reference). Inputs: codes/scales as produced by quant.py (q8 has sm baked, k8 centered),
    lse (B,H,N) natural, delta (B,H,N) in dP units (or None -> matched delta_norm computed here).
    Returns dq, dk, dv (B,N,H,D)/(B,N,Hk,D) fp32 and the delta used, rowsum(dS) residual."""
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    rep = H // Hk
    Q = _deq(q8, qs)
    K = _deq(k8, ks).repeat_interleave(rep, dim=1)
    V = _deq(v8, vs).repeat_interleave(rep, dim=1)
    dO = _deq(do8, dos)
    x = torch.matmul(Q, K.transpose(-1, -2)) * LOG2E
    if causal:
        mask = torch.ones(N, N, dtype=torch.bool, device=x.device).tril()
        x = x.masked_fill(~mask, float("-inf"))
    P = torch.exp2(x - (lse * LOG2E)[..., None])
    dP = torch.matmul(dO, V.transpose(-1, -2))
    if delta is None:
        delta = (P * dP).sum(-1) / P.sum(-1)
    dS = P * (dP - delta[..., None])
    rowsum = dS.sum(-1)
    PT = _q_blocks(P.transpose(-1, -2), 1, 32)          # per key, per 32-query block
    dSq = _q_blocks(dS, 128, 32)                         # per (128-query tile x 32 keys), shared by dK and dQ
    dV = torch.matmul(PT, dO)
    dK = torch.matmul(dSq.transpose(-1, -2), Q)
    dQ = torch.matmul(dSq, K) * sm
    if rep > 1:
        dK = dK.view(B, Hk, rep, N, D).sum(2)
        dV = dV.view(B, Hk, rep, N, D).sum(2)
    t = lambda a: a.permute(0, 2, 1, 3)  # noqa: E731
    return t(dQ), t(dK), t(dV), delta, rowsum


def cp_p_ref(q8, qs, k8, ks, causal):
    """sm100-mx-cp backward P (B,H,N,N): the forward's e4m3 codes * 2^(kb-7-m) / l_codes (rows sum to 1), and kb (B,H,N,N/32)."""
    B, N, H, D = q8.shape
    rep = H // k8.shape[2]
    Q = _deq(q8, qs)
    K = _deq(k8, ks).repeat_interleave(rep, dim=1)
    x = torch.matmul(Q, K.transpose(-1, -2)) * LOG2E
    if causal:
        x = x.masked_fill(~torch.ones(N, N, dtype=torch.bool, device=x.device).tril(), float("-inf"))
    xb = x.view(B, H, N, N // BLK, BLK)
    bmax = xb.amax(-1, keepdim=True)
    kb = torch.where(torch.isfinite(bmax), torch.floor(bmax), torch.zeros_like(bmax))
    pc = torch.exp2(xb - kb + 7).to(torch.float8_e4m3fn).float()
    m = kb.amax(-2, keepdim=True)
    P = (pc * torch.exp2(kb - 7 - m)).view(B, H, N, N)
    return P / P.sum(-1, keepdim=True), kb.squeeze(-1)


def bwd_ref_cp(q8, qs, k8, ks, v8, vs, do8, dos, causal, sm, P=None):
    """sm100-mx-cp backward (dev ref): P = forward code-P (cp_p_ref, or given), delta = sum P dP, then as bwd_ref."""
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    rep = H // Hk
    if P is None:
        P, _ = cp_p_ref(q8, qs, k8, ks, causal)
    Q = _deq(q8, qs)
    K = _deq(k8, ks).repeat_interleave(rep, dim=1)
    V = _deq(v8, vs).repeat_interleave(rep, dim=1)
    dO = _deq(do8, dos)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    delta = (P * dP).sum(-1) / P.sum(-1)
    dS = P * (dP - delta[..., None])
    PT = _q_blocks(P.transpose(-1, -2), 1, 32)
    dSq = _q_blocks(dS, 128, 32)
    dV = torch.matmul(PT, dO)
    dK = torch.matmul(dSq.transpose(-1, -2), Q)
    dQ = torch.matmul(dSq, K) * sm
    if rep > 1:
        dK = dK.view(B, Hk, rep, N, D).sum(2)
        dV = dV.view(B, Hk, rep, N, D).sum(2)
    t = lambda a: a.permute(0, 2, 1, 3)  # noqa: E731
    return t(dQ), t(dK), t(dV), delta, dP
