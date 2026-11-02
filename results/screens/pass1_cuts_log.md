# Round 19, worker S19-p1: pass 1 (FlashAttnBwdSm90Fp8R, D128) per-element / per-step instruction cuts on r18_c4

Base: kda-sm90 24275e3 (csrc = r18_c4). A = `.local/frozen/r18_c4`. Screen = `screen.sh` (2-rep smoke A/B, B2 N8192
H16/4 and H32/4, causal matched; noise about +-0.5..1%). Bitwise = `equivalence.py compare` on the default route (72
cases) vs the r17_pk byte reference (C4 is bitwise to it). Trees: `.local/wt/S19-p1-NN`. Libraries:
`.local/jit/S19-p1-NN/_C.so`. Patches: `.local/screens/S19-p1-NN/variant.patch` (diff vs pristine HEAD csrc, also in
`.local/rounds/r19-p1/patches/`). SASS counter: `.local/rounds/r19-p1/tools/p1count.py <_C.so>` (the unmasked EXPBX step
loop of the causal LPT kernel, with its conditional arms).

## SASS ledger of r18_c4 (unmasked step loop 0x7370-0xaf70, 961 static instructions)

Per thread per step (32 elements): exp 32 FFMA + 32 MUFU.EX2; dS line 32 FFMA + 32 FMUL; amax 32 FMNMX (serial chain
started from max(0, |x0|)); scale 32 FMUL; e4m3 16 + 16 F2FP (PACK_AB_MERGE_C). Per row: 16 FFMA (LSE bias) + 16 FMUL
(dPsum * k8s) + 16 LDS.64. dK fold 64 FFMA. Conditional arms:

| arm | static | runs on | executed per warp-step |
|---|---:|---|---:|
| dV rescale (rf != 1) | 64 FMUL | 1/2 of steps | 32 |
| dV promote (m & 7 == 7): 64 LDS + 64 FFMA + 64 STS + 64 zero MOV/IMAD + ~12 UMOV | 267 | 1/8 of steps | 33.4 |
| dS' TMA box issue (wg_thread 0) | 11 | 2 of 8 warps | 2.8 |
| psi STG with 64-bit address math (thread 0) | 23 | 1 of 8 warps | 2.9 |
| everything else | 596 | every step | 596 |
| **total** | 961 | | **~667** |

(The ledger's ~698 also counts retry stubs / per-step overhead outside this range; differences below are what matter.)

## Variants (all bitwise, all REG 168)

| tag | cut | executed instr / warp-step (delta vs r18_c4) | bitwise | pass1 H16 / H32 | STACK (P1R causal) | verdict |
|---|---|---:|---|---:|---|---|
| S19-p1-01 | DVFRESH: the first tile of each promote window (and of the CTA) runs the dV gemm with runtime scale-D = 0 (`tiled_mma_dKV.accumulate_ = Zero`), so the promote's 64 re-zeroing moves and that tile's rescale (of zeros) go | ~652 (**-15**: promote arm 267 -> 193, rescale on 3/8 of steps instead of 1/2) | EQUAL (72) | -0.33 / +0.02% | 0 | neutral |
| S19-p1-02 | 01 + BOOK: next q8s / do8s LDG (and its 64-bit address math) only when the next m-block starts a new 128-row scale block; psi row pointer hoisted out of the loop | ~646 (**-21**) | EQUAL | +0.19 / -0.94% | **8** (the hoisted pointer) | reject (stack, neutral) |
| S19-p1-03 | 02 + DVVEC: the dV overlay as float4 groups per thread, LDS.128 / STS.128 (promote arm 193 -> 97) | ~634 (**-33**) | EQUAL | -0.13 / -0.24% | 8 | reject (neutral, stack) |
| S19-p1-04 | 01 + PSILATE: psi publish moved from thread 0 (warp 0, before the dK issue, also the TMA issuer) to thread 32 after the dS' transpose (dK in flight) | ~651 (-16; moved, not cut) | EQUAL | **+1.32 / +1.59%** | 0 | reject (slower) |
| S19-p1-05 | 01 + ROWEARLY: the 16 LSE-bias FFMA (+LDS) before the S-retire wait and the 16 dPsum*k8s FMUL (+LDS) before the dP-retire wait | ~652 (-15; moved) | EQUAL | **+1.95 / +1.61%** | 0 | reject (slower) |
| S19-p1-06 | 01 + DVVEC (no BOOK) | ~640 (**-27**) | EQUAL | -0.16 / -1.04% | **8** (float4 overlay) | reject (neutral, stack) |

Reading: cutting 15..33 executed instructions per warp-step from the dV promote / rescale / scale-load bookkeeping moves
pass 1 by 0..-1% (noise). Those instructions sit in bubbles (the rescale fills the S-retire wait by design (S17
RESCALE-EARLY); the promote runs after the dK retire, overlapping the other warpgroup's gemms), so the ~0.05 sol4
points per instruction sensitivity does not apply to them; it holds only for the per-element chain between the S
retire and the dK issue (where SHFL8 / NOCLAMP paid). Re-placing per-row or per-thread work (04, 05) costs 1.3..2%:
the current placement is already the better schedule.

## Angles 1-3 (per-element, the chain that does count): counted, not built

- (1) dS = P^ (dP a - b) in one op: dS needs two products of P^ (P^ dP a and P^ b); a precomputed -P^ b is itself one
  op per element, so FFMA + FMUL is minimal (bitwise or not). Folding a into the exp (P' = 2^(y + log2 a)) changes P^
  (fails v4's per-row test, as EXPB did) and still needs P^ b. Saving: 0. Not built.
- (2) amax pass removal: sm90 has no 3-input FMNMX and no scaled e4m3 convert; a binary max over 32 values needs 31
  ops whatever the order (pairwise |a|,|b| FMNMX gives no saving); SATFINITE already clamps at 448. A bound-based psi
  (no amax): the only operand-known bound on |dP| is D * 448^2 * do8s * v8s (Cauchy-Schwarz on the codes), ~10^2..10^3
  above the realized amax, so most dS' codes would underflow; the milder variant (coarser codes) was measured by
  S18-nm-10 / DKACC: dQ / dK per-case error rises 2e-5..1.2e-3, FAIL. Saving 32 FMNMX (+~20 per-step reduction) but
  certain v4 failure. Not built.
- (3) packed f16x2 / bf16x2: per element pair the current line is FFMA 2 + FMUL 2 + FMNMX 2 + FMUL 2 + F2FP 1 + P~ F2FP 1 = 10.
  Packing only the scale (F2FP f16 pack + HMUL2 + cvt e4m3x2.f16x2 = 3 vs FMUL 2 + F2FP 1) saves 0. Packing the
  product as well (fp32 FFMA for t = dP a - b kept to avoid the dP - delta cancellation in 11 bits; t and P^ packed to
  f16x2, HMUL2, HMNMX2 |.|, HMUL2, cvt) = 9: saves 1 per pair = 16 per warp-step (~1.2% at the 0.05-point
  sensitivity) at best, with f16 range hazards (dS = P^ t is not bounded by 65504; inf amax zeroes the tile) and a
  changed psi (amax of rounded values) on every tile. Even a v4 pass could not reach 3%. Not built.
- (4) integer / control bookkeeping: 02 (LDG skip, pointer hoist) cut 6 more and added STACK 8; descriptor rebuilds
  (~4 uniform ops per gemm group from the smem base) are ptxas rematerialization under uniform-register pressure, not
  source-controllable without UNROLL2-style stage specialization (measured +8.9%, spills, by kda-pass1).
- (5) skipped-path instructions: DVFRESH (01) removes the dead rescale and re-zeroing; the remaining rescale arm is a
  uniform branch around 64 FMUL (nothing executed when skipped); the promote arm costs 1 branch when skipped.

## Verdict

No candidate. Every bitwise cut found in the emitted SASS (up to -33 executed instructions per warp-step, 03 / 06) is
neutral on pass 1 (best -1.04% on one shape, within noise; 02 / 03 / 06 also add STACK 8). The per-element chain
(exp FFMA + MUFU, dS FFMA + FMUL, amax, scale, e4m3) has no shorter bitwise form on sm90, and its non-bitwise
shortenings either change P^ / psi in ways already measured to fail residual-v4 (EXPB, DKACC) or are capped at ~1%
(packed product). DVFRESH (01, bitwise, STACK 0, -15 instructions) is the only clean piece; it measured -0.33 / +0.02%,
below the 1% stacking threshold.

Jobs: 21192803 (01), 21192804 (02), 21192829 (03), 21192948 (04), 21192947 (05), 21193143 (06); logs
`.local/logs/kda-S19-p1-NN-<job>.log`.
