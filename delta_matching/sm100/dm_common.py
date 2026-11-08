"""Arithmetic shared bit-for-bit by the matched-delta pass 0 and the main backward (kda-sm100).

Contract (sm100-mx backward): x = fma(S, c, -L2) with c = (ks * qs) * log2(e), L2 = lse * log2(e);  P = exp2(x).
Both kernels must call dm_bwd_p_pair so that P is bitwise identical (the tcgen05 fp8 MMA gives bitwise-equal S in
both orientations, see kda/docs/findings.md); any change of the exp implementation is made here, once.
"""
import math

import cutlass.cute as cute
from cutlass import Float32

LOG2E = math.log2(math.e)


def dm_bwd_c(ks: Float32, qs: Float32) -> Float32:
    """c = (ks * qs) * log2(e)  (this multiplication order)."""
    return (ks * qs) * Float32(LOG2E)


def dm_bwd_dp_scale(vs: Float32, dos: Float32) -> Float32:
    """a = vs * dos  (this multiplication order); dP = dP_acc * a."""
    return vs * dos


def dm_bwd_p_pair(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """(P0, P1) = exp2(fma((s0, s1), c, -l2)) with ex2.approx.ftz.f32."""
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return cute.math.exp2(x0, fastmath=True), cute.math.exp2(x1, fastmath=True)
