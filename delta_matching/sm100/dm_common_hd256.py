"""hd256 arithmetic shared bit-for-bit by the hd256 matched-delta pass 0 and the hd256 backward.

=== SPEC (USER DECISION 2026-11-10): MXFP8 = sm100-mx at D=256 (dm-validator preset "sm100-mx-d256") -> dm_mx256_* ===
The hd128 sm100-mx P / dP line VERBATIM (each helper calls the hd128 dm_common function, so hd128 and hd256 share one
implementation).  Inputs: Q / K / V / dO e4m3 codes with ONE pow2 (UE8M0) fp32 scale per 32 tokens over all of D
(quant.quant_rows rule), Q pre-multiplied by sm = 256^-0.5, K centered.  Per element (row i = query, col j = key):
  S_acc  = raw fp32 accumulator of tcgen05.mma kind::f8f6f4 (Q8 K8^T) over D=256 as 8 K=32 steps in increasing d
           (zero-init); causal: S_acc = -inf where j > i (on the raw accumulator)
  c      = (ks[j//32] * qs[i//32]) * log2(e)          dm_mx256_c(ks, qs)         FMUL, FMUL (this association)
  l2     = lse[i] * log2(e)                           dm_mx256_l2(lse)           FMUL RN (lse natural log, fp32)
  x      = fma(S_acc, c, -l2)                         fma.rn.f32x2 (per element = FFMA RN)
  P      = ex2.approx.ftz.f32(x)                      dm_mx256_p_pair(s0, s1, c, l2)   (no shift, no cap: hd128 line)
  a      = vs[j//32] * dos[i//32]                     dm_mx256_dp_scale(vs, dos) FMUL
  dP     = dP_acc * a        (dP_acc = raw f8f6f4 accumulator of dO8 V8^T, same instruction shape as S)
The scales are on the MMAs' non-contraction dims (rows / cols of S and dP) and are exact powers of two, so the
post-MMA multiply equals the kind::mxf8f6f4 block-scaled product with uniform SFs; like hd128's pass 0, no SF TMEM is used.
The backward must compute P with dm_mx256_p_pair from bitwise the same S_acc, c, l2 (and dP_acc, a).

=== RETIRED: [64,32] fp32 contract ("Delta-Matching-256 v1" floor) -> dm_p0h_* (kept only so older branches import) ===
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

from delta_matching.sm100 import dm_common

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


# ---------------------------------------------------------------------------------- MXFP8 (sm100-mx-d256) = THE spec
def dm_mx256_c(ks: Float32, qs: Float32) -> Float32:
    """c = (ks * qs) * log2(e)  (= dm_common.dm_bwd_c, this association)."""
    return dm_common.dm_bwd_c(ks, qs)


def dm_mx256_l2(lse: Float32) -> Float32:
    """l2 = lse * log2(e) (FMUL RN; = hd128 pass 0 / bwd)."""
    return lse * Float32(dm_common.LOG2E)


def dm_mx256_dp_scale(vs: Float32, dos: Float32) -> Float32:
    """a = vs * dos  (= dm_common.dm_bwd_dp_scale); dP = dP_acc * a."""
    return dm_common.dm_bwd_dp_scale(vs, dos)


def dm_mx256_p_pair(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """(P0, P1) = ex2.approx.ftz.f32(fma.rn((s0, s1), c, -l2))  (= dm_common.dm_bwd_p_pair)."""
    return dm_common.dm_bwd_p_pair(s0, s1, c, l2)
