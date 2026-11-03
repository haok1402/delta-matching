# Round 23, worker S23-pr: accuracy-preserving reformulations of pass 1's psi / dS' hand-off and the dV / dK promotes

Base: HEAD 64486b9 (csrc identical to e80de8f = r21_clean). A = `.local/frozen/r21_clean` in every screen.

Locations:
- Trees: `.local/wt/S23-pr-NN` (git archive of HEAD); DBG trees `.local/wt/S23-pr-NN-dbg` (benchmarks/v5/apply_v5_dbg.py).
- Libraries: `.local/jit/S23-pr-NN`, `.local/jit/S23-pr-NN-dbg`.
- Patches: `.local/screens/S23-pr-NN/variant.patch` (diff against the pristine HEAD archive `.local/rounds/r23-pr/pristine`; copies in
  `.local/rounds/r23-pr/patches/`).
- Tools: `.local/rounds/r23-pr/tools/` (mktree / edit.py + per-variant spec files sp_*.py / mkpatch / sub / screen / gate).
- Screen: `tools/screen.sh` = `.local/rounds/r22-fwd/screen.sh` (2-rep A/B vs r21_clean, B2 N8192 H16/4 and H32/4, D128 causal
  matched; bitwise = default-route `equivalence.py compare` against `.local/equiv4/r21clean_default.pt`).
- Gate (non-bitwise variants): `tools/gate.sh` = numerics.py production captures on 4 routes + `err_audit.py` vs the
  ORIGINAL (`base`), DBG build + DBG == production check, v5-abs (c = 1) on cases x 4 routes, stress and accept (3 seeds) vs
  the ORIGINAL's DBG captures (`.local/rounds/r21-v5/data/orig`; the original's accept capture regenerated in
  `.local/rounds/r23-pr/data/orig_acc`). Reports: `~/workspace/haok1402/kda-results/r23-pr/S23-pr-NN/`.

Build facts that matter for every rounding argument (from the r21_clean SASS of the unmasked step loop,
`.local/rounds/r23-pr/r21_loop.txt`): the library is built with `--use_fast_math`, so every fp32 op is `.FTZ`,
`1.0f / psi` is `MUFU.RCP` (approximate, not IEEE), `x / y` is `x * MUFU.RCP(y)`, and `amax * (1 / 448)` is an FMUL by
fl(1/448). Per element of the 64 x 128 tile (one thread holds 32): exp line FFMA + FADD(+8) + FMNMX(clamp, EXPB 0 keeps it
on every element) + MUFU.EX2; dS line FFMA (t = dP a - b) + FMUL (P^ t); amax FMNMX; after the psi barrier FMUL (x psi_inv)
+ 1/2 F2FP. Per row (16 per thread): LDS + FMUL (lse * log2e), LDS + FMUL (dPsum * k8s).

## Idea 1: exact power-of-two psi

Notation: x = one fp32 dS' element of the tile (k8s-folded units), A = the tile's max |x|, Q = e4m3 round-to-nearest with
SATFINITE (grid: subnormals k 2^-9, k = 1..7; normals (1 + m/8) 2^e, e = -6..8; max 448 = 1.75 2^8). The original:
psi_o = fl(A fl(1/448)), code = Q(fl(x rcp(psi_o))), reconstructed as psi_o code (pass 2) and g = psi_o q8sc / 256 (dK fold).
Snapped: psi_2 = 2^ceil(log2 psi_o) = r psi_o with r in [1, 2) (r = 1 iff psi_o is already a power of two).

Rounding argument (per element, reconstructed-value error e(x) = |psi Q(x / psi) - x|; the fp32 FMUL by an exact power of two
adds no rounding, the original's FMUL adds <= 2^-24 relative, three orders below Q):
- Normal band (|x| / psi in [2^-6, 448]): |e| <= 2^-4 |x| for both scales. The two grids are the same grid shifted by the
  factor r, so for a given x neither dominates; only the average over a smooth distribution is equal.
- Subnormal band (|x| < 2^-6 psi): |e| <= psi 2^-10, i.e. the bound is r times the original's (up to 2x), and the band itself
  is r times wider (more elements in it); the flush-to-zero threshold psi 2^-10 also moves up by r.
- The tile max: the original maps A to 448 exactly (error 0 up to the 2^-23 scale mismatch); the snapped scale maps it to
  A / psi_2 in (224, 448], an arbitrary point of the grid (error up to 2^-4 A).
- psi_o is the smallest scale with no saturation. Any psi' with no saturation has psi' >= psi_o, so its subnormal band is
  coarser; snapping DOWN (psi' < psi_o) saturates the largest elements (A / psi' up to 896 clamped to 448: up to 50% error).
  Hence no power-of-two choice can be provably no worse than the original: its bound is never better and is strictly worse
  in the subnormal band and at the max element whenever psi_o is not a power of two. This is why S17-psi-09 (round 17)
  measured many pairs worse (up to +1.4% relative): the dS distribution of a softmax-weighted tile has most elements far
  below the max (subnormal band), where the snapped grid is up to 2x coarser.
- What the snap buys: 1 / psi and every pass-2 ratio psi_w / psi_cur become exact powers of two (pass 2 can skip the ratio
  rescale whenever two consecutive taken tiles share the exponent), and the scale can be folded exactly (idea 2).

## Idea 2: fold the per-element scale into an existing FFMA

- The scale s = 1 / psi depends on the amax of the very products it scales, so it is not known when the dS line runs. The only
  FFMA upstream of the amax is the cancellation step t = fma(dP, a, -b) (dP a ~ delta on matched rows). Folding s there gives
  fma(dP, fl(a s), -fl(b s)): two input roundings ahead of the cancellation, whose error relative to t is amplified by
  (|dP a| + |b|) / |t|. The separate FMUL adds one rounding relative to the final value. So the fused form is strictly WORSE
  in bound unless s is a power of two (then fl(a s), fl(b s) are exact and the fused line is bitwise the power-of-two-psi
  line of idea 1, with the exponent predicted from the previous tile and an exact power-of-two correction on a miss).
- Downstream of the scale there is no FFMA: the e4m3 convert (F2FP.SATFINITE) has no scale operand on sm90.
- So idea 2 is either worse in bound (any non-power-of-two s) or idea 1. Its maximum pass-1 time value is the removal of the
  32 FMUL per thread-step = probe S23-pr-04 below.

## Idea 3: per-row / per-tile products computed once when exact

(3a) KFOLDSCALE (k8s out of the per-row line). Today: b = fl(dPsum k8s) per row (16 FMUL per thread-step) and
a_k = fl(fl(do8sc v8s) k8s) per step feed the cancellation FFMA t = fl(dP a_k - b); x = fl(P^ t); code = Q(fl(x psi_inv)).
New: t'' = fl(dP a - dPsum) with a = fl(do8sc v8s), x'' = fl(P^ t''), the warp amax multiplied by k8s once before the slot
write (fl(k max|x''|) = max fl(k |x''|) by monotonicity of rounding), code = Q(fl(x'' s_wg)) with s_wg = fl(psi_inv k8s)
(one scalar per step per WG). The codes keep the k8s fold, so psi / pass 2 / the dK fold / the exit unfold are unchanged in
meaning.
Bound (u = 2^-24, exact target X = P^ (dP a - delta) k / psi):
- original: |err| <= 4u |X| + u k P^ (|dP a| + |delta|) / psi (the last term: the two roundings of a_k and b ahead of the
  cancellation, amplified by it);
- new: |err| <= 5u |X| (one extra tile-uniform rounding, s_wg; no rounding ahead of the cancellation).
- Since P^ (|dP a| + |delta|) >= P^ |dP a - delta| = |X| psi / k, the new bound is <= the original's for every element, with
  equality only when there is no cancellation. Not bitwise: the published psi moves at the ulp level and codes flip where an
  element sits within a few ulps of an e4m3 rounding boundary.
(3b) lse * log2e per row (16 FMUL per thread-step): bitwise if fl(lse log2e) were read from a buffer (the pre-M17 host prep, or
pass 0 writing the value it already computes). Measured as a probe first (01).
(3c) per-step scalars (dsc, dp_descale, the F2b floor) are already one op per step; the floor's division is an FMUL by a
MUFU.RCP issued off the chain (SASS).

## Idea 4: power-of-two dV rescale factors / the promote

- rf = clamp(f_prev / f_cur), f = do8sc / 256: a power-of-two rf needs power-of-two dO block scales, i.e. a quantizer change
  that moves the dO codes, dP and P~ dO (not a reformulation of pass 1). The time value of any cheaper rescale is bounded by
  removing it entirely: kda-p1packed probe 1 measured that +2.5% SLOWER (the 64 FMUL fill the S-retire bubble). Not built.
- Promote: the bitwise reformulation DVFRESH (the first tile of a window starts the dV gemm with scale-D = 0; the window's
  rescale of zeros and the promote's re-zeroing go) re-screened on r21_clean (08). The window length is the numerics lever
  (PW32 / PT32 fail the per-case dV rule); a promote at every dO-scale change (no rescale at all, one fewer rounding per
  element) costs 4x the promotes (64 LDS + 64 FFMA + 64 STS each), which are smem-bandwidth bound: not built.

## Idea 5: cheaper exact dK fold

- tdK = fma(tmpK, g, tdK) is one correctly rounded op per element: the minimum for adding a scaled tile into an fp32 sum.
- With g a power of two (it is psi q8sc / 256; q8sc is a quantizer scale) the accumulator could be rescaled exactly and the
  dK gemm could accumulate in place, but then the fp8 wgmma accumulates across tiles (its internal accumulation is what the
  per-tile temp avoids): that is DKACC, measured failing (S18-nm-10, dQ / dK error +2e-5..1.2e-3). Pairwise pre-combination
  of two tiles adds roundings and keeps a 64-register temp live across steps (S18-nm-05: STACK 208, +29%). sm90 has no FFMA2.
  No cheaper exact form exists; not built.

## Variants

Screens are 2-rep, pass 1 at H16 / H32 (B2 N8192), percent B vs A (negative = faster). The noise is about +-1%, and in some
jobs the untouched kernels drift by up to 1.4% (columns "p0" / "fwd" show it). REG/STACK is the two Fp8R symbols
(non-causal, causal); r21_clean has 168/8, 168/0. No variant prints a ptxas spill or serialization diagnostic.

| tag | idea | change | numerics | pass1 H16 / H32 | other kernels in the same job | Fp8R REG/STACK | verdict |
|---|---|---|---|---:|---|---|---|
| S23-pr-01 | 3b | PROBE NOLSEMUL: no per-row lse * log2e FMUL (16 per thread-step) | probe (wrong) | **+1.43 / +2.20** | p0 0.00 / +0.46, fwd +0.07 / +0.46 | 168/8, 168/0 | the bound is >= 0: the bitwise lse_log2 buffer cannot pay. Not built |
| S23-pr-02 | 3a | PROBE NODPSK: no per-row dPsum * k8s FMUL (16 per thread-step) | probe (wrong) | -1.19 / -2.14 | p0 -0.61 / -1.45, fwd -0.79 / -1.36 | 168/8, 168/0 | the bound net of job drift is about -0.6 / -0.7% |
| S23-pr-04 | 1, 2 | PROBE: no psi multiply before the dS' convert (KDA_PK_PROBE 2; 32 FMUL per thread-step) | probe (wrong) | +0.21 / -0.55 | p0 -0.14 / -1.16, p2 -1.09 / -1.78 | 168/8, 168/0 | noise, as kda-p1packed's probe 2 on r17_p1: the scale multiply that power-of-two psi or a fused fold would remove is worth nothing in pass 1 |
| S23-pr-05 | 3 (psi) | PSITREE: the 8 warp-slot max as a depth-3 tree over two LDS.128 (was a 9-deep serial FMNMX chain after the barrier), and the per-thread 32-element amax as 4 chains | **bitwise** (72 default) | +0.69 / -0.52 | p0 +0.03 / -0.91 | 168/8, 168/0 | neutral. The SASS shows the tree; the post-barrier chain latency is not binding |
| S23-pr-06 | 3a | KFOLDSCALE: k8s out of the dS line (no dPsum * k8s per row, no a * k8s per step), the warp amax times k8s before the slot write, codes = Q(x'' fl(psi_inv k8s)) | not bitwise; gate below | +0.19 / +0.89 | p0 -0.02 / +0.45 | 168/8, 168/0 | **reject**: no time (the probe's ~0.6% does not materialize) and the dK mean rises by 5e-9 |
| S23-pr-07 | 1 | PSI2UP: psi snapped up to 2^ceil(log2 psi) by exponent bits, psi_inv the exact reciprocal power of two; pass 2 skips the ratio rescale when psi_w == psi_cur | not bitwise; gate below | +0.22 / +0.83; **pass 2 -0.79 / -0.77; pass 1 + pass 2 +0.02 / +0.52** | p0 -0.09 / +0.51 | 168/8, 168/0 | **reject**: FAIL error clause by 40-650x; no time |
| S23-pr-08 | 4 | DVFRESH (S19-p1-01) on r21_clean: zero-init dV gemm at a window start, no rescale of zeros, no re-zeroing | **bitwise** (72 default) | -0.31 / -1.05 | p0 +0.43 / -0.27 | 168/8, 168/0 | neutral (as on r18_c4: -0.33 / +0.02) |
| S23-pr-10 | stack | 05 + 08 (bitwise pieces) | **bitwise** (72 default) | +0.28 / -0.37 | p0 +0.16 / -0.29 | 168/8, 168/0 | neutral: nothing to hold (< 1%) |
| S23-pr-09 | stack | 05 + 06 + 08 | (06's numerics) | **+1.93 / +2.28** | p0 +0.17 / +0.21 | 168/8, **168/24** | reject: the causal Fp8R picks up STACK 24 and is slower |

Not built (argument above plus prior measurements): power-of-two scale folded into the dS FFMA (idea 2; its whole pass-1 value
is probe 04 = noise, and its numerics are 07's); power-of-two dV rescale (idea 4; needs power-of-two dO scales, and removing the
rescale entirely was +2.5% slower in kda-p1packed's probe 1); a cheaper exact dK fold (idea 5; none exists on sm90; DKACC fails
numerics, deferred folds spill); the bitwise lse_log2 buffer (probe 01 >= 0).

### Gate of the non-bitwise variants (S23-pr-06, S23-pr-07; tools/gate.sh, jobs 21209387 / 21209388, original's accept capture job 21209386)

Error clause, production path, against the ORIGINAL (`base` = 3d4cd5b) on 72 cases x 4 routes. The four routes are identical
for both. The 6 non-finite pairs are the original's d128_zeros / d256_zeros NaNs, as for r21_clean.

| variant | changed case-outputs | fail A (+1e-5 abs) | fail B (rel) | worst | mean rise O / dQ / dK / dV |
|---|---|---|---|---|---|
| S23-pr-06 KFOLDSCALE | 48 | 0 | 0 | d128_gqa3_promote/causal/matched dK 0.0636031 -> 0.0636034 (+2.9e-7, rel +4.6e-6); d128_mqa matched dK +2.6e-7 | 0 / **-9.1e-9** / **+5.0e-9** / 0 |
| S23-pr-07 PSI2UP | 60 | **52** | **52** | d128_outliers/full/stale dK 0.23653 -> 0.24304 (+6.5e-3, rel +2.8%); d128_scale10/full/stale dK +5.3e-3 and dQ +5.1e-3; matched d128_mqa dQ 0.06311 -> 0.06354; d128_gqa4/causal/stale dK 0.06466 -> 0.06558 (the same +1.4% as S17-psi-09) | 0 / **+2.8e-4** / **+3.7e-4** / 0 |

Residual-v5 (v5-abs, c = 1, every row vs the ORIGINAL): both variants 0 rows worse, max c needed 0, on cases x 4 routes
(334,592 rows each), stress (1,703,936) and accept (11,796,480 rows, 3 seeds). This is by construction: neither variant
touches P^, dP or delta, and the DBG rows are formed before the dS' line.

DBG == production: 07 yes (0 differing case-output errors). 06 NO (42 differ): with k8s gone, the dS line's dS * dp_descale is
the same expression as the DBG rows' product, so the DBG build CSEs it into FMUL + FADD instead of the production FFMA. A real
06 would need `__fmaf_rn` in the dS line. Not fixed, because 06 is rejected on time and on the dK mean.

Reports: `~/workspace/haok1402/kda-results/r23-pr/S23-pr-0{6,7}/` (err_vs_base.json / .log, abs_*.json / .log).
Captures: `.local/rounds/r23-pr/data/S23-pr-0{6,7}/` (the accept dumps and the original's regenerated accept dump were deleted).

### Reading

- Idea 1 (power-of-two psi): measured exactly as the argument predicts. The snapped grid is up to 2x coarser in the subnormal
  band where most dS elements of a softmax-weighted tile sit, and the max element loses its exact 448 code. The result is 52 of
  72 x 4 case-outputs over the clause, by up to 650x, with both means rising by 3-4e-4. It also buys no time: pass 1's psi
  multiply is free (probe 04), and pass 2's skipped ratio rescales are -0.8% of pass 2, i.e. pass 1 + pass 2 +0.02 / +0.52%.
  Snapping down saturates, so no power-of-two rule can be no worse.
- Idea 2 (fused scale): the only FFMA upstream of the amax is the cancellation step. Folding a non-power-of-two scale there is
  worse in bound; a power-of-two one is idea 1. Its time value is probe 04 = 0.
- Idea 3 (per-row / per-tile products): k8s out of the per-row line (06) is the one form with a strictly better bound. It
  passes the per-case clause easily (worst +2.9e-7). But it changes psi at the ulp level everywhere, so the dK mean moves by
  +5e-9 (a coin flip at that size; dQ moved -9e-9), and it is not faster (+0.2 / +0.9%). lse * log2e as a per-row input is
  >= 0 in time (probe 01). The post-barrier max tree (05) is bitwise and neutral.
- Idea 4 / 5 (promotes): the only reformulation with no numerics change, DVFRESH, is neutral again (08). Cheaper rescale forms
  are bounded by probe 1 (+2.5% slower when removed entirely). The dK fold is already one correctly rounded FFMA per element.
- Where pass 1's time is: the per-element chain that responds to instruction cuts is the exp phase. There MUFU.EX2 at 16 / clk / SM
  (2 consumer warps x 32 per SMSP = 512 cycles per step) and about 290 ALU instructions per warp share the issue port. The
  psi / dS' hand-off and the promotes sit under in-flight gemms (dV is issued before the psi barrier, dK before the
  transpose), so removing instructions there is noise (04, 05, 08, and S19-p1). The larger psi lever is structural, not a
  reformulation. Per-warp psi codes for dK only would make the dK issue barrier-free; it is bound-no-worse, like HALFPSI. Pass 2
  would keep the tile-psi codes, so dQ stays bitwise. It needs a second scale + convert (+48 instructions per thread-step) and
  keeping the 32 fp32 dS live past the dK issue (about 252 > 240 registers). Its ceiling is the WG-decoupling probe (FR16
  WGPSI -1.7%) minus that cost, so it was not built.

## Verdict

**No candidate.** I built and screened 9 variants (3 probes, 3 bitwise reformulations / stacks, 2 named numerics changes, 1
mixed stack) and gated the 2 non-bitwise ones with the residual-v5 tools:
- Exact power-of-two psi is provably not no-worse (its bound is never better), and it fails the error clause massively (52
  fails, means +3e-4).
- The one bound-better reformulation (KFOLDSCALE) passes per case but raises the dK mean by 5e-9 and gains nothing.
- The bitwise pieces (PSITREE, DVFRESH, their stack) are neutral (-1.05..+0.69%). Nothing reaches the 1% hold bar.

Jobs and logs (`.local/logs/kda-S23-pr-NN-<job>.log`): 01 21208966, 02 21208967, 04 21208968, 05 21208969, 06 21209039, 07 21209040,
08 21209041, 10 21209396, 09 21209457; gates 06 21209387, 07 21209388; original accept capture 21209386.
