"""hd256 ([64,32] fp32 block-scale contract, "Delta-Matching-256 v1" floor): arithmetic shared bit-for-bit by the
hd256 matched-delta pass 0 and the (future) hd256 backward.  hd128 uses dm_common.py and is NOT affected by this file.

Contract (spec kda/docs/sm90-numerics-spec-hd256.md §3 / §4.1 / §8.2), per element (row i = query, col j = key):
  S_acc  = raw fp32 accumulator of tcgen05.mma kind::f8f6f4 over D=256 as 8 K=32 steps in increasing d order
           (zero-init), masked entries set to -inf on the raw accumulator.
  dsc    = fmax((q8s[i//64] * k8s[j//32]) * log2(e), FLT_MIN)    (this association; FLT_MIN floor = spec §2.2/§8.2,
                                                                  bitwise-neutral for healthy scales)
  l2     = lse[i] * log2(e)                                        (FMUL RN; lse natural log, fp32)
  x      = FFMA(S_acc, dsc, -l2)        (fma_packed_f32x2 = FFMA2, per-element bitwise = scalar FFMA RN)
  x      = FADD(x, 8.0)
  x      = FMNMX(x, 12.0)               (min, NaN-quiet minimumNumber like fminf)
  P_hat  = MUFU.EX2(x)                  (ex2.approx.ftz.f32)          = 2^8 * exp(S - LSE), capped at 2^12
  dp     = do8s[i//64] * v8s[j//32]                                  (this association)
  dP     = dP_acc * dp                  (dP_acc = raw f8f6f4 accumulator of dO8 V8^T, same instruction shape as S)
The backward must compute P_hat with dm_p0h_phat_pair (same function) from bitwise the same S_acc, dsc, l2.
"""
import math

import cutlass.cute as cute
from cutlass import Float32

LOG2E = math.log2(math.e)
FLT_MIN = 1.1754943508222875e-38
PHAT_SHIFT = 8.0     # P_hat = 2^8 * P (spec "MO", +8 in exp2)
PHAT_CAP = 12.0      # spec F1 cap: exp2 argument <= 12
INV_PHAT = 1.0 / 256.0


def dm_p0h_dsc(qs: Float32, ks: Float32) -> Float32:
    """dsc = fmax((q8s * k8s) * log2(e), FLT_MIN)  (this association)."""
    return cute.arch.fmax((qs * ks) * Float32(LOG2E), Float32(FLT_MIN))


def dm_p0h_dp_scale(dos: Float32, vs: Float32) -> Float32:
    """dp = do8s * v8s  (this association); dP = dP_acc * dp."""
    return dos * vs


def dm_p0h_l2(lse: Float32) -> Float32:
    """l2 = lse * log2(e) (FMUL RN)."""
    return lse * Float32(LOG2E)


def dm_p0h_phat_pair(s0: Float32, s1: Float32, dsc: Float32, l2: Float32):
    """(P_hat0, P_hat1) = ex2.approx.ftz(fmin(fma((s0, s1), dsc, -l2) + 8, 12))."""
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (dsc, dsc), (-l2, -l2))
    x0, x1 = cute.arch.add_packed_f32x2((x0, x1), (Float32(PHAT_SHIFT), Float32(PHAT_SHIFT)))
    x0 = cute.arch.fmin(x0, Float32(PHAT_CAP))
    x1 = cute.arch.fmin(x1, Float32(PHAT_CAP))
    return cute.math.exp2(x0, approx=True, ftz=True), cute.math.exp2(x1, approx=True, ftz=True)
