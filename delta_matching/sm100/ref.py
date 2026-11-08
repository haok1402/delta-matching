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


def fwd_ref(q8, qs, k8, ks, v8, vs, causal, emulate_p=True):
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
    l = (pt.sum(-1, keepdim=True) * w).sum(-2).squeeze(-1)        # (B, H, N)
    P = (pc * w).view(B, H, N, N)
    O = torch.matmul(P, V) / l[..., None]
    lse = (m.squeeze(-1).squeeze(-1) + torch.log2(l)) / LOG2E
    return O.permute(0, 2, 1, 3), lse
