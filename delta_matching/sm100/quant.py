"""sm100-mx producers: e4m3 codes with one power-of-2 (UE8M0-representable) scale per 32 tokens over all of head_dim.

quant_rows(x, prescale, mean)  x (B, N, H, D) bf16/fp32 (any strides, unit stride on D)
    y = (x - mean[b, h]) * prescale            (fp32; mean optional, (B, H, D) fp32)
    per (b, h, 32-token block):  amax = max |y| = m * 2^E (m in [1, 2), fp32 bits);
        scale = the smallest power of 2 with amax <= 448 * scale  = 2^(E-8) if m <= 1.75 else 2^(E-7)
        (amax == 0 -> scale = 1);  code = rn_e4m3(y / scale)  (exact multiply by 2^-e; never exceeds 448)
    returns codes (B, N, H, D) float8_e4m3fn contiguous, scales (B, H, N // 32) fp32 (exact powers of 2)

The scale exponent biased by 127 is the UE8M0 byte used by the tcgen05 block-scaled MMAs.
"""
import torch
import triton
import triton.language as tl

BLK = 32


@triton.jit
def _quant_rows_kernel(X, MEAN, Y, S, N, H, sxb, sxn, sxh, prescale,
                       HAS_MEAN: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rows = pid_n * BLK + tl.arange(0, BLK)
    cols = tl.arange(0, D)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxn + cols[None, :]).to(tl.float32)
    if HAS_MEAN:
        mu = tl.load(MEAN + pid_bh * D + cols)
        x = x - mu[None, :]
    x = x * prescale
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    ab = amax.to(tl.int32, bitcast=True)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)   # smallest pow2 s: amax <= 448 s
    eb = tl.minimum(tl.maximum(eb, 1), 254)
    eb = tl.where(amax > 0, eb, 127)
    scale = (eb << 23).to(tl.float32, bitcast=True)
    inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    y = (x * inv).to(tl.float8e4nv)
    tl.store(Y + (b * N * H + rows[:, None] * H + h) * D + cols[None, :], y)
    tl.store(S + pid_bh * (N // BLK) + pid_n, scale)


def quant_rows(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    B, N, H, D = x.shape
    assert N % BLK == 0 and x.stride(-1) == 1
    y = torch.empty(B, N, H, D, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=x.device, dtype=torch.float32)
    m = mean.contiguous() if mean is not None else s
    _quant_rows_kernel[(N // BLK, B * H)](x, m, y, s, N, H, x.stride(0), x.stride(1), x.stride(2), float(prescale),
                                          HAS_MEAN=mean is not None, D=D, BLK=BLK, num_warps=4)
    return y, s


def kmean(k: torch.Tensor) -> torch.Tensor:
    """(B, N, Hk, D) -> (B, Hk, D) fp32 mean over tokens (K centering)."""
    return k.float().mean(dim=1)


def quant_rows_ref(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    """Pure-torch reference of quant_rows (bitwise target)."""
    B, N, H, D = x.shape
    y = x.float()
    if mean is not None:
        y = y - mean[:, None, :, :]
    y = y * prescale
    yb = y.view(B, N // BLK, BLK, H, D)
    amax = yb.abs().amax(dim=(2, 4))                              # (B, N/32, H)
    ab = amax.contiguous().view(torch.int32)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(torch.int32)
    eb = eb.clamp(min=1, max=254)
    eb = torch.where(amax > 0, eb, torch.full_like(eb, 127))
    scale = (eb << 23).view(torch.float32)
    inv = ((254 - eb) << 23).view(torch.float32)
    codes = (yb * inv[:, :, None, :, None]).to(torch.float8_e4m3fn).view(B, N, H, D)
    return codes, scale.permute(0, 2, 1).contiguous()
