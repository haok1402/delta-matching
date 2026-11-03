# S23-fr: accuracy-preserving reformulations of the forward's per-group basis work (r21_clean tile)

Base: kda-sm90 HEAD 64486b9 (= r21_clean sources: original 128 x 256 tile, 2 consumer WGs, 4 x 64-key groups per tile).
Trees `.local/wt/S23-fr-NN` (production) and `.local/wt/S23-fr-NN-dbg` (DM_DBG=1 + `benchmarks/v5/apply_v5_dbg.py`), both
`git archive HEAD`. Libraries `.local/jit/S23-fr-NN{,-dbg}`. Scripts `.local/rounds/r23-fr/` (`build.sh`, `capture.sh`,
`check.sh`, `screen.sh` = the r22 screen with BITWISE off and REPS, `scripts/edit.py` = the variant generator).
Reports `~/workspace/haok1402/kda-results/r23-fr/`. Patches `.local/screens/S23-fr-NN/variant.patch` (against HEAD).

Gates (vs the ORIGINAL): v5-abs at c = 1 (`numerics_v5.py check --form v5-abs`) against `.local/rounds/r21-v5/data/orig`
(cases x 4 routes, stress; accept regenerated from `.local/wt/v5-orig`); error clause with `err_audit.py` against
`r21-v5/data/base` (frozen original production): reading A (cand <= orig + 1e-5 per case / output) and the per-output mean
over cases not rising. Timing: `screen.sh` (2 reps, B2 N8192 H16/4 and H32/4) vs `.local/frozen/r21_clean`, then 4-rep.

## Where the per-element work is (r21_clean unmasked step, 256 keys, per warp; kda-fwd / S22 counts)

FMUL 411 = descale 128 + rho 3 x 64 + alpha 64; FFMA 212 = code 128 + promote 64; 1273 instructions per step. FA3 has
the code FFMA and alpha, not the descale, rho or promote.

## Idea 1: fold the S descale into the code FFMA (removes the 128 descale FMUL)

Original: S = RN(x g) (g = q8sc k8sc_g >= FLT_MIN), group max on S, ms_g = RN(fmax(RN(gm_g l2), floor) - c) (c = log2 448),
code argument a = RN(S l2 - ms_g) (one FFMA).

Rounding argument.
- Maxima: RN is monotone and g > 0, so max_j RN(x_j g) = RN(max_j x_j * g). The group maxima, row_max, alpha and the floor
  computed from RN(gm_raw * g) once per row and group are BITWISE the original's (exact equivalence, not a bound).
- Code argument, errors against the exact a* = x g l2 - gm g l2 + c (X = x g l2, M = gm g l2, u = 2^-24):
  - original: X d1 (the RN(x g)) + M (da + db) (RN(gm g), RN(. l2)) + c-subtraction + final FFMA rounding:
    bound u (|X| + 2|M|) + u |ms| + u |a|.
  - 1a, plain fold, a = RN(x RN(g l2) - ms_g) with ms_g unchanged: X d2 + M (da + db) + ...: the SAME bound (the per-element
    product rounding moves into one per-group rounding of g l2). Equal bound, not per-element dominance.
  - 1b, fold + basis from the same product, ms_g = RN(fmax(RN(gm_raw RN(g l2)), floor) - c): the rounding of g l2 now
    multiplies X - M only, because the codes and the basis use the same multiplier: (X - M) d2 + M db + ...:
    bound u (|X - M| + |M|) + u |ms| + u |a| <= u (|X| + 2|M|) + ...: PROVABLY NO WORSE than the original's bound, and
    strictly smaller for the elements that carry the weight (X near M: u |M| against up to 3 u |M|).
- What the bound does not cover: the LSE. row_sum is formed from the pre-cast codes, so it changes at fp32 round-off,
  and the backward recomputes P^ with the ORIGINAL arithmetic (RN(s dsc) l2 - lse). In the original the per-element
  rounding d1 is common to the forward's row_sum and the backward's P^ and cancels in sum P^ - 2^8; with the fold it does
  not, so the normalization residual moves by up to ~2^-23 |S l2| per element. v5-abs allows k 2^-23 S_r per row, so the
  fold is applied on UNMASKED steps only (C3g's choice): masked (diagonal) steps keep the exact original arithmetic, so
  every row with fewer than 256 visible keys is bitwise the original's in the forward.

## Idea 2: exact power-of-two O basis (lazy reference snapped to integers; alpha folded into the promote)

O kept in 2^(-E) units with integer E: the cross-tile rescale is exact (power of two) and skipped while E is unchanged, the
promote weight becomes v8sc_3 2^(ms_3 - E), finalize multiplies by 2^(E - m l2) / l.
Rounding argument: NOT provably no worse. The denominator must keep the computed alpha (row_sum and LSE are the
backward's contract), and in the original the SAME computed alpha multiplies numerator and denominator, so its ex2.approx
error cancels in O = num / l. With an exact numerator rescale the denominator's alpha errors no longer cancel (one term
per row-max change), and the final 2^(E - m l2) is an extra ex2.approx rounding on the whole row. Same class as FWDLAZY,
whose BN256 port (kda-fwd e256nf) raised the O mean against the original (results/rounds.md round 21). Not built.

## Idea 3: snap a per-group basis or rho to a power of two

rho_g = (v8sc_{g-1} / v8sc_g) 2^(ms_{g-1} - ms_g) is fixed by the codes' basis ms_g and the V scales. Making it a power of
two needs ms_g moved by a non-integer amount, which changes the e4m3 codes non-dyadically: a convention change (the
shared-basis FWDSHB class, O error +1.3e-3 .. 4.2e-3; the header's 2.77e-2 vs dyadic 1.4e-7 measurement). A dyadic snap of
ms_g alone (integer shift) leaves rho non-dyadic through v8sc ratios and ms differences. Rejected by argument, not built.

## Idea 4: per-group promote (O += D_g w_g, each 64-key group in its own zero-init temp; no rho chain)

Element ops: 4 FFMA instead of 3 FMUL + 1 FFMA (equal); per-row scalars: 3 weights (FADD, EX2, FMUL) instead of 3 rho
(FADD, EX2, FMUL, 2 clamps, plus the V-ratio RCP / FMUL): about -12 instructions per step.
Rounding argument: the original chain feeds each group's rescaled partial back into the fp8 wgmma as its accumulator, so
the tensor core's limited-precision accumulate (~14 effective bits, relative to the largest addend) is applied to the
multi-group partial at every later k-atom (6 atom boundaries per tile carry earlier groups), plus three fp32 FMUL
roundings of the partial. Per-group: the tensor core only accumulates the second atom of a group onto the first (same
basis), and the combination is done by fp32 FFMA into O (one 2^-24 |O| rounding each, three more per tile than the
original). Bound: original ~2^-14 sum |partial| over 6 boundaries + 3 x 2^-24 |D| + 2^-24 |O|; per-group ~2^-14 |D_g| x 4
+ 4 x 2^-24 |O|. No worse unless |O| >> 2^10 |tile partial| (rows of 2^18+ keys); for the gate's lengths (<= 64K keys,
<= 256 tiles) not worse in bound. Role: instruction-neutral; an accuracy bank to stack with idea 1.

## Finding on idea 1: the backward already uses the folded factor

Pass 0 (`fp11_delta_v6.h` dsc = fq8_q8sc * k8s * log2e) and pass 1 (`mainloop_bwd_sm90_fp8r.hpp:779`
fq8_s_descale_log2 = q8sc * k8s_wg * log2e) compute P^ = exp2(fma(s, RN(RN(q8sc k8s) log2e), -lse l2) + 8). So the
ORIGINAL forward (RN(x g), then * l2) is the one that disagrees with the backward. S23-fr-01 uses exactly the backward's
factor (gsc = RN(q8sc k8sc), gl2 = RN(gsc l2), l2 = log2e), which makes the forward's codes / row_sum use the same
per-element scaled score as the backward. That is a better argument than the forward-only bound, but the gate below still
fails. It differs from kda-fwd's FOLDC, whose factor is RN(RN(q l2) k), not the backward's.

## Variants (default route, 72 cases; timing 2-rep vs r21_clean, B2 N8192 D128 causal matched)

Error columns are from err_audit.py vs the original (frozen base). "A fails" counts case / output pairs over +1e-5.
Mean rises are per output: O / dQ / dK / dV, and a positive value fails. The 4 routes give identical errors in every
earlier audit, so the default route decides.

| tag | change | v5-abs c=1 (cases default) | A fails (worst) | mean rise O / dQ / dK / dV | fwd H16 / H32 | verdict |
|---|---|---|---|---|---|---|
| S23-fr-01 | idea 1a: fold, backward's factor, unmasked steps only (FWDDFOLD=1 on the BN256 tile, no lazy) | PASS (0 rows, max c 0.215) | 1 (scale10 full/stale dQ +1.06e-5) | -9.3e-10 / **+3.4e-7** / -1.5e-7 / -2.6e-9 | -2.79% / -3.04% | FAIL (A, dQ mean) |
| S23-fr-02 | idea 1b: 01 + the basis ms_g from the same folded product (the provably-no-worse bound) | PASS (0, 0.215) | 2 (scale10 full/stale dQ +4.27e-5, dK +4.15e-5) | -1.9e-8 / -1.2e-6 / -1.3e-6 / **+1.1e-8** | -3.79% / -4.35% | FAIL (A, dV mean) |
| S23-fr-03 | 01 on every step (masked + prologue too) | **FAIL** (4 rows, c 3.43; short causal rows) | 4 | +1.2e-9 / +1.2e-6 / +5.9e-7 / -1.4e-8 | -4.37% / -3.99% | FAIL |
| S23-fr-04 | 02 on every step | **FAIL** (6 rows, c 3.43) | 2 | +8.6e-9 / -1.5e-7 / -3.4e-7 / +1.0e-8 | -4.13% / -3.96% | FAIL |
| S23-fr-05 | idea 4: per-group promote (no rho chain) | PASS (rows bitwise: LSE unchanged) | 11 (mqa causal/stale dK +2.5e-4; scale3 full/stale dQ +2.0e-4) | **+6.6e-7 / +3.4e-6 / +6.0e-6** / 0 | -1.79% / -1.69% | FAIL |
| S23-fr-06 | 02 + 05 | PASS (0, 0.215) | 11 | +6.4e-7 / +3.1e-6 / +5.5e-6 / +1.1e-8 | -3.27% / -2.65% | FAIL |
| S23-fr-07 | bitwise scalar: the row-sum fold's sigma_3 reused as the promote weight's exp (-2 EX2 / FADD per step) | - | - | - | +0.25% / +0.56% | reject: not bitwise on d256 (d256_gqa4 stale dQ; the d256 promote reads basis_last at another point), and no gain on hd128 |

Every variant compiles at REG 168 / STACK 24 (r21_clean's), with no spill. Accept and stress were not run, because no
variant passes the default-route error clause, which is the screen.

## Reading

- The fold family (01-04) buys 3-4.4% forward, as much as the 128 FMUL predicts. The O error mean even drops for the
  better-bounded form (02: -1.9e-8). But the changed exp arguments flip a few e4m3 codes at rounding boundaries. Those
  flips move the stale scale10 / scale3 gradients chaotically: +-4e-5 .. 9e-5 per case in both directions (02 improves
  scale10 causal/stale dQ by 9.2e-5 and worsens full/stale by 4.3e-5). The per-output means then land on either side of
  zero at the 1e-8 .. 1e-7 level. A better rounding bound does not control which way a code flips, so the literal
  clause (no case +1e-5, no mean rise) is a lottery for this family. This matches kda-fwd's FOLDC / FOLDK.
- Folding the masked (diagonal) steps too fails v5-abs on short causal rows (c 3.43), even with the backward-consistent
  factor. Unmasked-only is required.
- Per-group promote (05) is instruction-neutral, yet measured -1.7% (fewer rho scalars / clamps). But the tensor-core
  accumulation argument does not show up against the fp32 reference. The O error moves +-3e-6 per case and the mean
  rises 6.6e-7, while the stale cases amplify it to +2.5e-4. Rejected.
- Ideas 2 and 3 are rejected by argument (above). The element work left (3 x 64 rho FMUL, 64 promote FFMA) is the
  4-basis floor: combining 4 differently-scaled products needs 3 element ops plus the promote, unless the bases are
  shared (a code change, the FWDSHB class) or the tensor core does the scaling (not on sm90).

## Verdict

No candidate. No variant passes v5-abs plus the literal error clause on the default route, so none went on to the four
routes, acceptance or stress. The fastest failing forms are 02 (-3.8 / -4.4%; misses dV mean by +1.1e-8 and fails 2
stale full scale10 pairs) and 03 / 04 (-4%; also fail v5).
