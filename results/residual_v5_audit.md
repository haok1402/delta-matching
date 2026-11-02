# r21-v5: residual-v5 measurement and re-audit of the accepted numerics changes (STAGING; integrator reviews / commits)

All files are in `.local/rounds/r21-v5/`. Check reports (JSON + log per pair and suite) are in
`~/workspace/haok1402/kda-results/r21-v5/<cand>_vs_<ref>/{cases_default,cases_kv,cases_head,cases_streams2,accept,stress}.json`.
Jobs: builds 21203146/7, 21203291-4. Captures 21203290, 21203295, 21203445-51, 21203504/5/7. Checks 21203383,
21203511-3, 21203701/2.

## Verdict

| accepted numerics change | pairs measured | v5 result | rows worse (72 cases, per route / acceptance 11.8M rows / stress 1.70M rows) |
|---|---|---|---|
| forward C3g (c1ceece, BM192 + FWDLAZY + FWDDFOLD) | c3g vs its predecessor 9be3622 (= original rows bitwise) | **FAILS v5** | 138 / 0 / 66,540 |
| EXPBX64 (7a06674) | exp vs its predecessor 19c238b | **FAILS v5** | 63 / 0 / 50,090 |
| C4 masked-line clamp + SHFL8 (65d4424) | c4 vs 7a06674 | **PASSES v5** (every row bitwise equal to its predecessor's, all suites, stress included) | 0 / 0 / 0 |
| cumulative: HEAD (= r19_fwd source) vs ORIGINAL 3d4cd5b | head vs orig | **FAILS v5** | 191 / 0 / 104,715 |
| fallback C3n2 (f96a26b, C3 without the fold) | c3n2 vs 9be3622 | FAILS v5 | 50 / 0 / 34,497 |
| fallback nofold (a8ee6dc = HEAD with FWDDFOLD=0) | nofold vs orig | FAILS v5 | 119 / 0 / 86,573 |

No other named numerics change appears in `results/rounds.md`. Every accepted bitwise interval is bitwise on every row of
every suite, stress included: orig -> 9be3622, c1ceece -> 19c238b, 65d4424 -> HEAD (0 worse, all rows equal). So the
cumulative failure comes entirely from C3g plus EXPBX64.

Worst rows (ratio original -> candidate, k = visible keys, den = sum |t|):
- C3g: d128_scale10 bh3 row 277 (k 278) 0.0534 -> 1.0 (den 0.082 -> 0.247, |delta| 23.2); d128_outliers bh0 row 752
  (k 753) 0.459 -> 1.0 (den 0.032 -> 0.0087, |delta| 94); stress outliers/s1 bh1 row 3329 0.0021 -> 1.0.
- EXPBX64: d128_outliers bh0 row 663 (k 664) 0.584 -> 0.984 (den 7.2e-5 -> 1.9e-3, |delta| 5.1); d128_scale10 bh3
  row 989 0.366 -> 1.0; d128_scale3 bh0 row 77 (k 78) 0.0075 -> 0.464; stress scale10/s1 bh15 row 2138: 0 (den 0) -> 1.0.
- Cumulative: the same rows; the maximum excess is 0.947 on the cases and 1.0 on stress.

**Read this before acting on the verdict: controls show the v5 predicate is below the original kernel's own round-off
noise.** Every failing row is a cancelling row: den = sum|t| is about 1e-4 .. 1e-2 of 2^8 |delta| (P near one-hot,
dP_j close to delta), so its terms are pure rounding residue and its ratio is O(0.01 .. 1) already in the original.
- Determinism: the original DBG run twice gives identical rows on every suite (orig2 vs orig: 0 worse, all rows equal).
- The ORIGINAL kernel with only its saved LSE moved by +1 / -1 ulp (Python-only control, same library; the r17
  `_lsep` / `_lsem` method) FAILS v5 against itself: 4,279 / 4,471 rows on the 72 cases, 165 / 155 rows on the
  acceptance shapes, and 375,854 / 453,431 rows on stress. That is 20-30x more than HEAD on the cases, and the
  acceptance shapes fail too, where HEAD has 0.
- More rows improve than get worse. HEAD vs orig: 463 better vs 191 worse on the cases, 180,909 vs 104,715 on
  stress, and the per-case median ratio drops in every changed case. C3g: 368 vs 138 better. C3n2: 40 vs 50.
- Rows worse beyond the original's own LSE +-1 ulp envelope (diagnostic only; v5 has no envelope): HEAD 1 row on the
  cases (d128_outliers), 0 on acceptance, 11,313 on stress (9,278 of them long_onehot2). C3g alone: 0 on the cases,
  6,535 on stress.

So under v5 as written, no non-bitwise change can pass, including a 1-ulp LSE change of the original kernel itself.
Reverting C3g and EXPBX64 is what rule 4 requires as written. Before that, Hao should rule on whether the cancelling
rows need a noise-aware form, e.g. an original-envelope allowance or a denominator that includes 2^8 |delta|. That
would be a rule question, and I have not applied it here. The fallbacks do not fix this: C3n2 and nofold still fail,
and nofold vs HEAD makes 358 rows worse (more than it fixes).

## (a) Exact tensors measured

The new DBG op is `torch.ops.delta_matching.bwd_fp8_t_dbg5`. It exists only in DM_DBG=1 builds and takes the
production `bwd_fp8_t` arguments. Per backward group call it returns `(dq, dk, dv, rows, rowsabs, cks, ds8, psi, rowsx)`.
- `rows`, `rowsabs` [BH, nbn, 2, s_q] fp32. These are the existing tensors, unchanged. They hold the per-(n-tile,
  warpgroup) fixed-order partials of t_rj = P^_rj (dP_rj - delta_r) and |t_rj|, where:
  - P^ = 2^8 exp(S - lse) is pass 1's fp32 exp output (`scores`), before the e4m3 conversion of P;
  - dP = fl(dP_acc * do8s * v8s) is pass 1's fp32 dP;
  - delta is the delta argument.
  So they are the fp32 dS terms before the k8s fold and before the e4m3 quantization of dS'. The rows are
  POST-exp and PRE-quantization, which is the quantity the identity sum_j dS_rj = 0 is defined on. The kernel code is in
  `mainloop_bwd_sm90_fp8r.hpp` (hd128) and `mainloop_bwd_sm90_fp8v10.hpp` (d256) and is identical in 3d4cd5b and HEAD.
  num_r = |sum t| and den_r = sum |t| are reduced in fp64 on the GPU.
- `rowsx` [3, BH, nbn, 2, s_q] fp64 (NEW). It is built from the SAME fp32 P^ and dP values: exact fp64 products,
  fp64 sums, the same 8-lane then 4-warp order. The planes are:
  - plane 0: sum P^ dP;
  - plane 1: sum P^;
  - plane 2: sum |P^ dP|.
  delta_recomputed = 2^-8 * plane 0, which is pass 0's matched definition (fp11_delta_v6.h `rowsum_step`: fp32 fma
  chains of P^ * fl(dP_acc * do8s * v8s) per 8 keys, x 2^-8, then an fp64 fold; pass 0 uses the same dp descale
  do8s * v8s). delta_used = the launch's delta argument (args[5]).
- The capture keeps, per matched row: num, den, sum P^dP, sum P^, sum |P^dP| (fp64), delta_used (fp32). k_r comes from
  the geometry. ds8 / psi / partials are dropped on the GPU right after each group (at 64K the capture forces the
  per-head split).
- Delta used vs recomputed: max |du - dr| / (2^-8 sum |P^ dP|) is at most 3.5e-7 on every case of every build. That is
  fp32 summation round-off, the same in the original and the candidates. Pass 0's delta and pass 1's P^ / dP still use
  the same products after EXPBX64 / C4. The residual comes from normalization instead: max |2^-8 sum P^ - 1| is about
  1e-6 on Gaussian cases and up to 5.7e-5 on scale10 (LSE vs recomputed exp). The degenerate zeros cases are excluded.
- Production is untouched. The extension is inside `#ifdef FA3FP11_DBG` (kernels) and the DBG op section (host). Before /
  after check: the production builds of the PATCHED original tree and the PATCHED HEAD tree are SASS-identical to the
  frozen unpatched libraries (`sass_cmp.py`: base 22/22 functions identical, r19_fwd 24/24 identical). Those frozen
  libraries are `.local/frozen/base` (from 3d4cd5b) and `.local/frozen/r19_fwd`, whose source = HEAD csrc.
- The DBG launches reproduce production. On all 72 cases the gradients and O of the v5 DBG builds are byte-identical to
  production captures of the frozen libraries (orig DBG vs base: 72/72; HEAD DBG vs r19_fwd: 72/72).

## (b) Staging tool and tests

- `apply_v5_dbg.py <tree> [--api-from <api.cpp>]` patches a source tree by anchors, with count assertions. It works
  unchanged on every tree from 3d4cd5b to HEAD. For the ORIGINAL tree, `--api-from` first installs HEAD's host file
  (round-13 twin-taking DBG op; that file is unchanged since d0af2af).
- `numerics_v5.py`:
  - `capture --root <tree> --suite cases|accept|stress [--route R]`. The cases suite runs the 72 cases through the
    repo's `benchmarks/equivalence.py` on all 4 routes. The accept suite runs B2 N8192 and B1 N65536 at 16/4 and 32/4,
    D128 causal matched, seeds 0/1/2. The stress suite runs scale3, scale10, outliers and near-one-hot (alpha 2, 1) at
    B1 H16/4 N8192, plus long N65536 scale3 / onehot2 at H4/1.
  - `check --orig --cand [--env lsep lsem]`. The predicate: per row, ratio_cand <= ratio_orig + k_r 2^-23, with NO
    exemptions; den = 0 requires num = 0; layouts and case sets must match. Each report gives per case: rows,
    candidate / original max and median, rows worse, rows better, rows worse among v4-exempt rows, max excess, delta
    used vs recomputed, the normalization error, and the 5 worst rows. `--env` adds the diagnostic count of rows worse
    beyond the original's own LSE +-1 ulp envelope; it never relaxes the predicate.
- `test_numerics_v5.py` holds 12 CPU tests, all passing:
  - equal rows pass; better rows pass; a rise within round-off passes;
  - a rise beyond round-off fails;
  - a degenerate k = 1 row (exempt under v2-v4) that got worse fails;
  - an ill-conditioned row (exempt under v4) that got worse fails;
  - zero-den with a nonzero num fails;
  - a layout mismatch or a missing case fails;
  - the delta report; the GPU reduction; the envelope stays diagnostic only.
- Scripts and data:
  - `build.sh`, `capture_all.sh`, `check_pair.sh`, `summarize.py`.
  - Trees are `.local/wt/v5-{orig,head,pc3g,c3g,c3n2,pexp,exp,c4,nofold}` (patched DBG), `v5-{orig,head}-prod`, and
    the controls `v5-{orig2,lsep,lsem}`. Libraries are in `.local/jit/v5-*`.
  - Captures are in `data/<tag>/`. The accept dumps (519 MB each) were deleted; `capture_all.sh <tag> accept`
    regenerates them in about a minute.

## (c) Audit tables

Per-pair totals (72 cases: identical on all 4 routes; worse / better):

| pair | cases rows worse / better | accept (11,796,480 rows) worse | stress (1,703,936 rows) worse / better |
|---|---|---|---|
| orig repeat vs orig (determinism) | 0 / 0 | 0 | 0 / 0 |
| orig LSE+1ulp vs orig (control) | 4,279 / 1,369 | 165 | 375,854 / 155,639 |
| orig LSE-1ulp vs orig (control) | 4,471 / 850 | 155 | 453,431 / 66,356 |
| HEAD vs orig (cumulative) | 191 / 463 | 0 | 104,715 / 180,909 |
| pre-C3g 9be3622 vs orig | 0 (bitwise) | 0 | 0 |
| C3g c1ceece vs 9be3622 | 138 / 368 | 0 | 66,540 / 112,984 |
| C3n2 f96a26b vs 9be3622 | 50 / 40 | 0 | 34,497 / 11,228 |
| pre-EXPBX64 19c238b vs c1ceece | 0 (bitwise) | 0 | 0 |
| EXPBX64 7a06674 vs 19c238b | 63 (worse) | 0 | 50,090 (worse) |
| C4 65d4424 vs 7a06674 | 0 (bitwise) | 0 | 0 |
| HEAD vs 65d4424 | 0 (bitwise) | 0 | 0 |
| nofold a8ee6dc vs orig | 119 / 144 | 0 | 86,573 / 85,089 |
| nofold vs HEAD | 358 (worse) | 0 | 133,620 (worse) |

Changed cases only. All other matched cases have 0 rows worse, and the d256 rows are bitwise the original's:

| case | rows | HEAD worse | C3g worse | EXPBX64 worse | C3n2 worse | nofold vs orig worse |
|---|---|---|---|---|---|---|
| d128_outliers | 4096 | 69 | 38 | 40 | 14 | 57 |
| d128_scale10 | 4096 | 80 | 76 | 4 | 20 | 24 |
| d128_scale3 | 4096 | 42 | 24 | 19 | 16 | 38 |

Cumulative HEAD vs the original, every case: `tables_head_vs_orig.md`. Its columns are rows, candidate / original
max and median ratio, worse, worse among v4-exempt rows, max excess, outputs equal, delta used vs recomputed
(candidate / original), and the normalization error. It covers the default route (the kv / head / streams2 tables are
identical), acceptance, and stress. Acceptance summary: 12 shape x seed cases, all PASS. Medians are 9.0e-9 (64K) and
1.12e-8 (8K) vs the original's 9.1e-9 / 1.15e-8. Max delta deviation is 0.9-1.6e-7 relative, equal to the original's.

#### cases_default: FAIL (191 of 334592 rows worse than the original)

| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d128_bhnd/causal/matched | 8192 | 1.17e-06 | 1.7e-08 | 1.17e-06 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 1.01e-07 / 1.01e-07 | 1.05e-06 |
| d128_constk/causal/matched | 4096 | 5.3e-07 | 1.37e-08 | 5.3e-07 | 1.26e-08 | 0 | 0 | -1.19e-07 | False | 5.23e-08 / 5.23e-08 | 9.07e-07 |
| d128_gqa3_promote/causal/matched | 7680 | 1.89e-06 | 1.53e-08 | 1.89e-06 | 1.65e-08 | 0 | 0 | -1.19e-07 | False | 7.73e-08 / 7.73e-08 | 9.47e-07 |
| d128_gqa4/causal/matched | 32768 | 1 | 1.37e-08 | 1 | 1.45e-08 | 0 | 0 | -1.19e-07 | False | 1.26e-07 / 1.26e-07 | 1.19e-06 |
| d128_heldout/causal/matched | 110592 | 1 | 1.3e-08 | 1 | 1.37e-08 | 0 | 0 | -1.19e-07 | False | 1.49e-07 / 1.51e-07 | 1.33e-06 |
| d128_min/causal/matched | 1024 | 1 | 2.76e-08 | 1 | 2.89e-08 | 0 | 0 | -1.19e-07 | False | 7.69e-08 / 7.69e-08 | 9.8e-07 |
| d128_mqa/causal/matched | 8192 | 1 | 1.63e-08 | 1 | 1.74e-08 | 0 | 0 | -1.19e-07 | False | 8.42e-08 / 8.66e-08 | 9.99e-07 |
| d128_outliers/causal/matched | 4096 | 1 | 3.87e-08 | 1 | 4.4e-08 | 69 | 33 | 0.541 | False | 2.75e-07 / 2.75e-07 | 6.95e-06 |
| d128_packed/causal/matched | 8192 | 1 | 1.68e-08 | 1 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 7.95e-08 / 7.95e-08 | 9.86e-07 |
| d128_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 80 | 70 | 0.947 | False | 1.08e-07 / 1.21e-07 | 5.7e-05 |
| d128_scale3/causal/matched | 4096 | 1 | 1.52e-06 | 1 | 1.99e-06 | 42 | 8 | 0.457 | False | 2.31e-07 / 2.32e-07 | 5.9e-06 |
| d128_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | False | 0 / 0 | 2.03e+03 |
| d256_bhnd/causal/matched | 8192 | 1 | 1.8e-08 | 1 | 1.8e-08 | 0 | 0 | -1.19e-07 | True | 1.05e-07 / 1.05e-07 | 1.13e-06 |
| d256_constk/causal/matched | 4096 | 5.2e-07 | 1.29e-08 | 5.2e-07 | 1.29e-08 | 0 | 0 | -1.19e-07 | True | 5.84e-08 / 5.84e-08 | 6.62e-07 |
| d256_gqa3_promote/causal/matched | 3840 | 1.98e-06 | 2.13e-08 | 1.98e-06 | 2.13e-08 | 0 | 0 | -1.19e-07 | True | 7e-08 / 7e-08 | 1.11e-06 |
| d256_gqa4/causal/matched | 32768 | 2.9e-06 | 1.5e-08 | 2.9e-06 | 1.5e-08 | 0 | 0 | -1.19e-07 | True | 6.95e-08 / 6.95e-08 | 1.28e-06 |
| d256_heldout/causal/matched | 55296 | 1 | 1.38e-08 | 1 | 1.38e-08 | 0 | 0 | -1.19e-07 | True | 1.15e-07 / 1.15e-07 | 1.52e-06 |
| d256_min/causal/matched | 512 | 1 | 3.26e-08 | 1 | 3.26e-08 | 0 | 0 | -1.19e-07 | True | 7.73e-08 / 7.73e-08 | 9.01e-07 |
| d256_mqa/causal/matched | 8192 | 2.65e-06 | 1.79e-08 | 2.65e-06 | 1.79e-08 | 0 | 0 | -1.19e-07 | True | 8.74e-08 / 8.74e-08 | 1.2e-06 |
| d256_outliers/causal/matched | 4096 | 1 | 3.22e-07 | 1 | 3.22e-07 | 0 | 0 | -1.19e-07 | True | 2.46e-07 / 2.46e-07 | 2.85e-05 |
| d256_packed/causal/matched | 8192 | 1 | 1.82e-08 | 1 | 1.82e-08 | 0 | 0 | -1.19e-07 | True | 9.01e-08 / 9.01e-08 | 1.19e-06 |
| d256_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 0 | 0 | -1.19e-07 | True | 1.14e-07 / 1.14e-07 | 4.76e-05 |
| d256_scale3/causal/matched | 4096 | 1 | 1.87e-06 | 1 | 1.87e-06 | 0 | 0 | -1.19e-07 | True | 1.86e-07 / 1.86e-07 | 5.82e-06 |
| d256_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | True | 0 / 0 | 1.64e+04 |

#### cases_kv: FAIL (191 of 334592 rows worse than the original)

| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d128_bhnd/causal/matched | 8192 | 1.17e-06 | 1.7e-08 | 1.17e-06 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 1.01e-07 / 1.01e-07 | 1.05e-06 |
| d128_constk/causal/matched | 4096 | 5.3e-07 | 1.37e-08 | 5.3e-07 | 1.26e-08 | 0 | 0 | -1.19e-07 | False | 5.23e-08 / 5.23e-08 | 9.07e-07 |
| d128_gqa3_promote/causal/matched | 7680 | 1.89e-06 | 1.53e-08 | 1.89e-06 | 1.65e-08 | 0 | 0 | -1.19e-07 | False | 7.73e-08 / 7.73e-08 | 9.47e-07 |
| d128_gqa4/causal/matched | 32768 | 1 | 1.37e-08 | 1 | 1.45e-08 | 0 | 0 | -1.19e-07 | False | 1.26e-07 / 1.26e-07 | 1.19e-06 |
| d128_heldout/causal/matched | 110592 | 1 | 1.3e-08 | 1 | 1.37e-08 | 0 | 0 | -1.19e-07 | False | 1.49e-07 / 1.51e-07 | 1.33e-06 |
| d128_min/causal/matched | 1024 | 1 | 2.76e-08 | 1 | 2.89e-08 | 0 | 0 | -1.19e-07 | False | 7.69e-08 / 7.69e-08 | 9.8e-07 |
| d128_mqa/causal/matched | 8192 | 1 | 1.63e-08 | 1 | 1.74e-08 | 0 | 0 | -1.19e-07 | False | 8.42e-08 / 8.66e-08 | 9.99e-07 |
| d128_outliers/causal/matched | 4096 | 1 | 3.87e-08 | 1 | 4.4e-08 | 69 | 33 | 0.541 | False | 2.75e-07 / 2.75e-07 | 6.95e-06 |
| d128_packed/causal/matched | 8192 | 1 | 1.68e-08 | 1 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 7.95e-08 / 7.95e-08 | 9.86e-07 |
| d128_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 80 | 70 | 0.947 | False | 1.08e-07 / 1.21e-07 | 5.7e-05 |
| d128_scale3/causal/matched | 4096 | 1 | 1.52e-06 | 1 | 1.99e-06 | 42 | 8 | 0.457 | False | 2.31e-07 / 2.32e-07 | 5.9e-06 |
| d128_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | False | 0 / 0 | 2.03e+03 |
| d256_bhnd/causal/matched | 8192 | 1 | 1.8e-08 | 1 | 1.8e-08 | 0 | 0 | -1.19e-07 | True | 1.05e-07 / 1.05e-07 | 1.13e-06 |
| d256_constk/causal/matched | 4096 | 5.2e-07 | 1.29e-08 | 5.2e-07 | 1.29e-08 | 0 | 0 | -1.19e-07 | True | 5.84e-08 / 5.84e-08 | 6.62e-07 |
| d256_gqa3_promote/causal/matched | 3840 | 1.98e-06 | 2.13e-08 | 1.98e-06 | 2.13e-08 | 0 | 0 | -1.19e-07 | True | 7e-08 / 7e-08 | 1.11e-06 |
| d256_gqa4/causal/matched | 32768 | 2.9e-06 | 1.5e-08 | 2.9e-06 | 1.5e-08 | 0 | 0 | -1.19e-07 | True | 6.95e-08 / 6.95e-08 | 1.28e-06 |
| d256_heldout/causal/matched | 55296 | 1 | 1.38e-08 | 1 | 1.38e-08 | 0 | 0 | -1.19e-07 | True | 1.15e-07 / 1.15e-07 | 1.52e-06 |
| d256_min/causal/matched | 512 | 1 | 3.26e-08 | 1 | 3.26e-08 | 0 | 0 | -1.19e-07 | True | 7.73e-08 / 7.73e-08 | 9.01e-07 |
| d256_mqa/causal/matched | 8192 | 2.65e-06 | 1.79e-08 | 2.65e-06 | 1.79e-08 | 0 | 0 | -1.19e-07 | True | 8.74e-08 / 8.74e-08 | 1.2e-06 |
| d256_outliers/causal/matched | 4096 | 1 | 3.22e-07 | 1 | 3.22e-07 | 0 | 0 | -1.19e-07 | True | 2.46e-07 / 2.46e-07 | 2.85e-05 |
| d256_packed/causal/matched | 8192 | 1 | 1.82e-08 | 1 | 1.82e-08 | 0 | 0 | -1.19e-07 | True | 9.01e-08 / 9.01e-08 | 1.19e-06 |
| d256_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 0 | 0 | -1.19e-07 | True | 1.14e-07 / 1.14e-07 | 4.76e-05 |
| d256_scale3/causal/matched | 4096 | 1 | 1.87e-06 | 1 | 1.87e-06 | 0 | 0 | -1.19e-07 | True | 1.86e-07 / 1.86e-07 | 5.82e-06 |
| d256_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | True | 0 / 0 | 1.64e+04 |

#### cases_head: FAIL (191 of 334592 rows worse than the original)

| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d128_bhnd/causal/matched | 8192 | 1.17e-06 | 1.7e-08 | 1.17e-06 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 1.01e-07 / 1.01e-07 | 1.05e-06 |
| d128_constk/causal/matched | 4096 | 5.3e-07 | 1.37e-08 | 5.3e-07 | 1.26e-08 | 0 | 0 | -1.19e-07 | False | 5.23e-08 / 5.23e-08 | 9.07e-07 |
| d128_gqa3_promote/causal/matched | 7680 | 1.89e-06 | 1.53e-08 | 1.89e-06 | 1.65e-08 | 0 | 0 | -1.19e-07 | False | 7.73e-08 / 7.73e-08 | 9.47e-07 |
| d128_gqa4/causal/matched | 32768 | 1 | 1.37e-08 | 1 | 1.45e-08 | 0 | 0 | -1.19e-07 | False | 1.26e-07 / 1.26e-07 | 1.19e-06 |
| d128_heldout/causal/matched | 110592 | 1 | 1.3e-08 | 1 | 1.37e-08 | 0 | 0 | -1.19e-07 | False | 1.49e-07 / 1.51e-07 | 1.33e-06 |
| d128_min/causal/matched | 1024 | 1 | 2.76e-08 | 1 | 2.89e-08 | 0 | 0 | -1.19e-07 | False | 7.69e-08 / 7.69e-08 | 9.8e-07 |
| d128_mqa/causal/matched | 8192 | 1 | 1.63e-08 | 1 | 1.74e-08 | 0 | 0 | -1.19e-07 | False | 8.42e-08 / 8.66e-08 | 9.99e-07 |
| d128_outliers/causal/matched | 4096 | 1 | 3.87e-08 | 1 | 4.4e-08 | 69 | 33 | 0.541 | False | 2.75e-07 / 2.75e-07 | 6.95e-06 |
| d128_packed/causal/matched | 8192 | 1 | 1.68e-08 | 1 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 7.95e-08 / 7.95e-08 | 9.86e-07 |
| d128_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 80 | 70 | 0.947 | False | 1.08e-07 / 1.21e-07 | 5.7e-05 |
| d128_scale3/causal/matched | 4096 | 1 | 1.52e-06 | 1 | 1.99e-06 | 42 | 8 | 0.457 | False | 2.31e-07 / 2.32e-07 | 5.9e-06 |
| d128_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | False | 0 / 0 | 2.03e+03 |
| d256_bhnd/causal/matched | 8192 | 1 | 1.8e-08 | 1 | 1.8e-08 | 0 | 0 | -1.19e-07 | True | 1.05e-07 / 1.05e-07 | 1.13e-06 |
| d256_constk/causal/matched | 4096 | 5.2e-07 | 1.29e-08 | 5.2e-07 | 1.29e-08 | 0 | 0 | -1.19e-07 | True | 5.84e-08 / 5.84e-08 | 6.62e-07 |
| d256_gqa3_promote/causal/matched | 3840 | 1.98e-06 | 2.13e-08 | 1.98e-06 | 2.13e-08 | 0 | 0 | -1.19e-07 | True | 7e-08 / 7e-08 | 1.11e-06 |
| d256_gqa4/causal/matched | 32768 | 2.9e-06 | 1.5e-08 | 2.9e-06 | 1.5e-08 | 0 | 0 | -1.19e-07 | True | 6.95e-08 / 6.95e-08 | 1.28e-06 |
| d256_heldout/causal/matched | 55296 | 1 | 1.38e-08 | 1 | 1.38e-08 | 0 | 0 | -1.19e-07 | True | 1.15e-07 / 1.15e-07 | 1.52e-06 |
| d256_min/causal/matched | 512 | 1 | 3.26e-08 | 1 | 3.26e-08 | 0 | 0 | -1.19e-07 | True | 7.73e-08 / 7.73e-08 | 9.01e-07 |
| d256_mqa/causal/matched | 8192 | 2.65e-06 | 1.79e-08 | 2.65e-06 | 1.79e-08 | 0 | 0 | -1.19e-07 | True | 8.74e-08 / 8.74e-08 | 1.2e-06 |
| d256_outliers/causal/matched | 4096 | 1 | 3.22e-07 | 1 | 3.22e-07 | 0 | 0 | -1.19e-07 | True | 2.46e-07 / 2.46e-07 | 2.85e-05 |
| d256_packed/causal/matched | 8192 | 1 | 1.82e-08 | 1 | 1.82e-08 | 0 | 0 | -1.19e-07 | True | 9.01e-08 / 9.01e-08 | 1.19e-06 |
| d256_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 0 | 0 | -1.19e-07 | True | 1.14e-07 / 1.14e-07 | 4.76e-05 |
| d256_scale3/causal/matched | 4096 | 1 | 1.87e-06 | 1 | 1.87e-06 | 0 | 0 | -1.19e-07 | True | 1.86e-07 / 1.86e-07 | 5.82e-06 |
| d256_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | True | 0 / 0 | 1.64e+04 |

#### cases_streams2: FAIL (191 of 334592 rows worse than the original)

| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d128_bhnd/causal/matched | 8192 | 1.17e-06 | 1.7e-08 | 1.17e-06 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 1.01e-07 / 1.01e-07 | 1.05e-06 |
| d128_constk/causal/matched | 4096 | 5.3e-07 | 1.37e-08 | 5.3e-07 | 1.26e-08 | 0 | 0 | -1.19e-07 | False | 5.23e-08 / 5.23e-08 | 9.07e-07 |
| d128_gqa3_promote/causal/matched | 7680 | 1.89e-06 | 1.53e-08 | 1.89e-06 | 1.65e-08 | 0 | 0 | -1.19e-07 | False | 7.73e-08 / 7.73e-08 | 9.47e-07 |
| d128_gqa4/causal/matched | 32768 | 1 | 1.37e-08 | 1 | 1.45e-08 | 0 | 0 | -1.19e-07 | False | 1.26e-07 / 1.26e-07 | 1.19e-06 |
| d128_heldout/causal/matched | 110592 | 1 | 1.3e-08 | 1 | 1.37e-08 | 0 | 0 | -1.19e-07 | False | 1.49e-07 / 1.51e-07 | 1.33e-06 |
| d128_min/causal/matched | 1024 | 1 | 2.76e-08 | 1 | 2.89e-08 | 0 | 0 | -1.19e-07 | False | 7.69e-08 / 7.69e-08 | 9.8e-07 |
| d128_mqa/causal/matched | 8192 | 1 | 1.63e-08 | 1 | 1.74e-08 | 0 | 0 | -1.19e-07 | False | 8.42e-08 / 8.66e-08 | 9.99e-07 |
| d128_outliers/causal/matched | 4096 | 1 | 3.87e-08 | 1 | 4.4e-08 | 69 | 33 | 0.541 | False | 2.75e-07 / 2.75e-07 | 6.95e-06 |
| d128_packed/causal/matched | 8192 | 1 | 1.68e-08 | 1 | 1.81e-08 | 0 | 0 | -1.19e-07 | False | 7.95e-08 / 7.95e-08 | 9.86e-07 |
| d128_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 80 | 70 | 0.947 | False | 1.08e-07 / 1.21e-07 | 5.7e-05 |
| d128_scale3/causal/matched | 4096 | 1 | 1.52e-06 | 1 | 1.99e-06 | 42 | 8 | 0.457 | False | 2.31e-07 / 2.32e-07 | 5.9e-06 |
| d128_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | False | 0 / 0 | 2.03e+03 |
| d256_bhnd/causal/matched | 8192 | 1 | 1.8e-08 | 1 | 1.8e-08 | 0 | 0 | -1.19e-07 | True | 1.05e-07 / 1.05e-07 | 1.13e-06 |
| d256_constk/causal/matched | 4096 | 5.2e-07 | 1.29e-08 | 5.2e-07 | 1.29e-08 | 0 | 0 | -1.19e-07 | True | 5.84e-08 / 5.84e-08 | 6.62e-07 |
| d256_gqa3_promote/causal/matched | 3840 | 1.98e-06 | 2.13e-08 | 1.98e-06 | 2.13e-08 | 0 | 0 | -1.19e-07 | True | 7e-08 / 7e-08 | 1.11e-06 |
| d256_gqa4/causal/matched | 32768 | 2.9e-06 | 1.5e-08 | 2.9e-06 | 1.5e-08 | 0 | 0 | -1.19e-07 | True | 6.95e-08 / 6.95e-08 | 1.28e-06 |
| d256_heldout/causal/matched | 55296 | 1 | 1.38e-08 | 1 | 1.38e-08 | 0 | 0 | -1.19e-07 | True | 1.15e-07 / 1.15e-07 | 1.52e-06 |
| d256_min/causal/matched | 512 | 1 | 3.26e-08 | 1 | 3.26e-08 | 0 | 0 | -1.19e-07 | True | 7.73e-08 / 7.73e-08 | 9.01e-07 |
| d256_mqa/causal/matched | 8192 | 2.65e-06 | 1.79e-08 | 2.65e-06 | 1.79e-08 | 0 | 0 | -1.19e-07 | True | 8.74e-08 / 8.74e-08 | 1.2e-06 |
| d256_outliers/causal/matched | 4096 | 1 | 3.22e-07 | 1 | 3.22e-07 | 0 | 0 | -1.19e-07 | True | 2.46e-07 / 2.46e-07 | 2.85e-05 |
| d256_packed/causal/matched | 8192 | 1 | 1.82e-08 | 1 | 1.82e-08 | 0 | 0 | -1.19e-07 | True | 9.01e-08 / 9.01e-08 | 1.19e-06 |
| d256_scale10/causal/matched | 4096 | 1 | 1 | 1 | 1 | 0 | 0 | -1.19e-07 | True | 1.14e-07 / 1.14e-07 | 4.76e-05 |
| d256_scale3/causal/matched | 4096 | 1 | 1.87e-06 | 1 | 1.87e-06 | 0 | 0 | -1.19e-07 | True | 1.86e-07 / 1.86e-07 | 5.82e-06 |
| d256_zeros/causal/matched | 4096 | 0 | 0 | 0 | 0 | 0 | 0 | -1.19e-07 | True | 0 / 0 | 1.64e+04 |

#### accept: PASS (0 of 11796480 rows worse than the original)

| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1n64k_16_4/s0 | 1048576 | 1 | 8.98e-09 | 1 | 9.12e-09 | 0 | 0 | -1.19e-07 | False | 1.2e-07 / 1.2e-07 | 2.1e-06 |
| b1n64k_16_4/s1 | 1048576 | 1 | 8.99e-09 | 1 | 9.11e-09 | 0 | 0 | -1.19e-07 | False | 9.53e-08 / 9.53e-08 | 2.35e-06 |
| b1n64k_16_4/s2 | 1048576 | 1 | 9e-09 | 1 | 9.12e-09 | 0 | 0 | -1.19e-07 | False | 1.59e-07 / 1.59e-07 | 2.13e-06 |
| b1n64k_32_4/s0 | 2097152 | 1 | 9e-09 | 1 | 9.13e-09 | 0 | 0 | -1.19e-07 | False | 9.14e-08 / 9.18e-08 | 2.24e-06 |
| b1n64k_32_4/s1 | 2097152 | 2.93e-05 | 9.02e-09 | 2.93e-05 | 9.13e-09 | 0 | 0 | -1.19e-07 | False | 1.09e-07 / 1.09e-07 | 2.21e-06 |
| b1n64k_32_4/s2 | 2097152 | 1 | 8.98e-09 | 1 | 9.1e-09 | 0 | 0 | -1.19e-07 | False | 8.87e-08 / 8.87e-08 | 2.26e-06 |
| b2n8k_16_4/s0 | 262144 | 1 | 1.12e-08 | 1 | 1.15e-08 | 0 | 0 | -1.19e-07 | False | 1.16e-07 / 1.16e-07 | 1.64e-06 |
| b2n8k_16_4/s1 | 262144 | 4.64e-06 | 1.12e-08 | 4.64e-06 | 1.15e-08 | 0 | 0 | -1.19e-07 | False | 1.03e-07 / 1.03e-07 | 1.71e-06 |
| b2n8k_16_4/s2 | 262144 | 1 | 1.12e-08 | 1 | 1.15e-08 | 0 | 0 | -1.19e-07 | False | 1.06e-07 / 1.06e-07 | 1.62e-06 |
| b2n8k_32_4/s0 | 524288 | 1 | 1.12e-08 | 1 | 1.16e-08 | 0 | 0 | -1.19e-07 | False | 1.29e-07 / 1.29e-07 | 1.61e-06 |
| b2n8k_32_4/s1 | 524288 | 1 | 1.11e-08 | 1 | 1.15e-08 | 0 | 0 | -1.19e-07 | False | 1.08e-07 / 1.08e-07 | 1.66e-06 |
| b2n8k_32_4/s2 | 524288 | 1 | 1.12e-08 | 1 | 1.15e-08 | 0 | 0 | -1.19e-07 | False | 1.41e-07 / 1.41e-07 | 1.67e-06 |

#### stress: FAIL (104715 of 1703936 rows worse than the original)

| case | rows | cand max | cand median | orig max | orig median | worse | worse (v4-exempt) | max excess | outputs = orig | max abs(du - dr) / (2^-8 sum abs(P dP)) cand / orig | max abs(2^-8 sum P - 1) cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long_onehot2/s0 | 262144 | 1 | 0.0106 | 1 | 0.0082 | 51175 | 47637 | 0.994 | False | 3.07e-07 / 3.07e-07 | 1.24e-05 |
| long_scale3/s0 | 262144 | 1 | 1.17e-06 | 1 | 1.31e-06 | 267 | 200 | 0.734 | False | 2.03e-07 / 2.12e-07 | 6.86e-06 |
| onehot1/s0 | 131072 | 1 | 1.33e-06 | 1 | 1.7e-06 | 802 | 21 | 0.0087 | False | 3.4e-07 / 3.53e-07 | 2.5e-06 |
| onehot2/s0 | 131072 | 1 | 0.0381 | 1 | 0.0538 | 18492 | 17834 | 0.997 | False | 3.27e-07 / 3.27e-07 | 4.84e-06 |
| onehot2/s1 | 131072 | 1 | 0.0359 | 1 | 0.0502 | 18462 | 17762 | 0.997 | False | 3.33e-07 / 3.33e-07 | 3.99e-06 |
| outliers/s0 | 131072 | 1 | 1.3e-05 | 1 | 1.83e-05 | 3996 | 3215 | 0.992 | False | 3.34e-07 / 2.86e-07 | 0.000178 |
| outliers/s1 | 131072 | 1 | 1.1e-05 | 1 | 1.5e-05 | 3965 | 3151 | 0.989 | False | 3.21e-07 / 3.05e-07 | 0.000179 |
| scale10/s0 | 131072 | 1 | 1 | 1 | 1 | 3398 | 3231 | 0.969 | False | 1.43e-07 / 1.43e-07 | 7.41e-05 |
| scale10/s1 | 131072 | 1 | 1 | 1 | 1 | 3452 | 3300 | 1 | False | 1.44e-07 / 1.44e-07 | 6.08e-05 |
| scale3/s0 | 131072 | 1 | 1.23e-06 | 1 | 1.46e-06 | 357 | 189 | 0.891 | False | 2.51e-07 / 2.51e-07 | 6.26e-06 |
| scale3/s1 | 131072 | 1 | 1.24e-06 | 1 | 1.47e-06 | 349 | 174 | 0.849 | False | 2.47e-07 / 2.47e-07 | 5.96e-06 |

## (e) v5-abs (integrator request): round-off of computing each row's sum from its own terms

**Predicate.** Per matched row, against the ORIGINAL, with no exemptions:

|sum t|_cand <= |sum t|_orig + c * k_r * 2^-23 * S_r,  where  S_r = sum_j |P^ dP| + |delta_used| * sum_j P^.

S_r is taken from the original's row. The factors were checked against the kernel. t = P^ (dP - delta), and P^ already
carries the 2^8 factor (P^ = 2^8 exp(S - lse)) in both `rows` and `rowsx`; dP and delta are in true dP units. So the
two products of t are P^ dP (rowsx plane 2 = sum |P^ dP|) and P^ delta (|delta_used| * plane 1 = sum P^). S_r is in t's
units as written, with no 2^+-8 rescaling. The fp32 partial sums of rows / rowsabs are formed from exactly these products.

**Code.** `numerics_v5.py check --form v5-abs` makes v5-abs the pass criterion. Every report carries both forms:
`abs_worse_c1`, `abs_c_needed` (the minimum c per case = max over rows of (|sum t|_cand - |sum t|_orig) / (k 2^-23 S_r),
infinite for a rise on a row with S = 0) and `abs_worst`. The reports are in
`~/workspace/haok1402/kda-results/r21-v5/<pair>/abs_<suite>.json` (jobs 21204034/5).

**Tests.** 7 new CPU tests, 19 in total, all passing. They cover:
- the S_r formula;
- equal rows pass;
- a rise within c = 1 passes even when the ratio form fails;
- a rise beyond the bound fails and reports c = 2;
- a degenerate k = 1 row has no exemption;
- a rise on a row with S = 0 needs infinite c;
- the total c is the max over cases, not a sum. My first aggregation summed it; the per-case values were right, and the
  JSON totals have been corrected.

| pair | suite | rows | v5 (ratio) worse | v5-abs worse at c = 1 | max c needed | case of max c |
|---|---|---|---|---|---|---|
| (a) orig LSE+1ulp vs orig | cases_default | 334592 | 4279 | 427 | 88.5 | d256_scale10/causal/matched |
| (a) orig LSE+1ulp vs orig | cases_kv | 334592 | 4279 | 427 | 88.5 | d256_scale10/causal/matched |
| (a) orig LSE+1ulp vs orig | cases_head | 334592 | 4279 | 427 | 88.5 | d256_scale10/causal/matched |
| (a) orig LSE+1ulp vs orig | cases_streams2 | 334592 | 4279 | 427 | 88.5 | d256_scale10/causal/matched |
| (a) orig LSE+1ulp vs orig | accept | 11796480 | 165 | 40 | 1.48 | b2n8k_32_4/s1 |
| (a) orig LSE+1ulp vs orig | stress | 1703936 | 375854 | 2349 | 88.7 | scale10/s1 |
| (b) orig LSE-1ulp vs orig | cases_default | 334592 | 4471 | 550 | 88.5 | d256_scale10/causal/matched |
| (b) orig LSE-1ulp vs orig | cases_kv | 334592 | 4471 | 550 | 88.5 | d256_scale10/causal/matched |
| (b) orig LSE-1ulp vs orig | cases_head | 334592 | 4471 | 550 | 88.5 | d256_scale10/causal/matched |
| (b) orig LSE-1ulp vs orig | cases_streams2 | 334592 | 4471 | 550 | 88.5 | d256_scale10/causal/matched |
| (b) orig LSE-1ulp vs orig | accept | 11796480 | 155 | 5 | 3.15 | b2n8k_32_4/s2 |
| (b) orig LSE-1ulp vs orig | stress | 1703936 | 453431 | 2025 | 65.7 | scale10/s0 |
| (c) HEAD vs orig | cases_default | 334592 | 191 | 0 | 0.954 | d128_scale10/causal/matched |
| (c) HEAD vs orig | cases_kv | 334592 | 191 | 0 | 0.954 | d128_scale10/causal/matched |
| (c) HEAD vs orig | cases_head | 334592 | 191 | 0 | 0.954 | d128_scale10/causal/matched |
| (c) HEAD vs orig | cases_streams2 | 334592 | 191 | 0 | 0.954 | d128_scale10/causal/matched |
| (c) HEAD vs orig | accept | 11796480 | 0 | 0 | 0.0119 | b1n64k_32_4/s0 |
| (c) HEAD vs orig | stress | 1703936 | 104715 | 0 | 0.616 | scale10/s0 |
| (d) C3g vs pre-C3g 9be3622 | cases_default | 334592 | 138 | 0 | 0.954 | d128_scale10/causal/matched |
| (d) C3g vs pre-C3g 9be3622 | cases_kv | 334592 | 138 | 0 | 0.954 | d128_scale10/causal/matched |
| (d) C3g vs pre-C3g 9be3622 | cases_head | 334592 | 138 | 0 | 0.954 | d128_scale10/causal/matched |
| (d) C3g vs pre-C3g 9be3622 | cases_streams2 | 334592 | 138 | 0 | 0.954 | d128_scale10/causal/matched |
| (d) C3g vs pre-C3g 9be3622 | accept | 11796480 | 0 | 0 | 0.0109 | b1n64k_32_4/s1 |
| (d) C3g vs pre-C3g 9be3622 | stress | 1703936 | 66540 | 0 | 0.598 | scale10/s1 |
| (e) EXPBX64 vs pre-EXPBX64 19c238b | cases_default | 334592 | 63 | 0 | 0.334 | d128_scale10/causal/matched |
| (e) EXPBX64 vs pre-EXPBX64 19c238b | cases_kv | 334592 | 63 | 0 | 0.334 | d128_scale10/causal/matched |
| (e) EXPBX64 vs pre-EXPBX64 19c238b | cases_head | 334592 | 63 | 0 | 0.334 | d128_scale10/causal/matched |
| (e) EXPBX64 vs pre-EXPBX64 19c238b | cases_streams2 | 334592 | 63 | 0 | 0.334 | d128_scale10/causal/matched |
| (e) EXPBX64 vs pre-EXPBX64 19c238b | accept | 11796480 | 0 | 0 | 0.0119 | b1n64k_32_4/s0 |
| (e) EXPBX64 vs pre-EXPBX64 19c238b | stress | 1703936 | 50090 | 0 | 0.616 | scale10/s0 |
| (f) nofold a8ee6dc vs orig | cases_default | 334592 | 119 | 0 | 0.954 | d128_scale10/causal/matched |
| (f) nofold a8ee6dc vs orig | cases_kv | 334592 | 119 | 0 | 0.954 | d128_scale10/causal/matched |
| (f) nofold a8ee6dc vs orig | cases_head | 334592 | 119 | 0 | 0.954 | d128_scale10/causal/matched |
| (f) nofold a8ee6dc vs orig | cases_streams2 | 334592 | 119 | 0 | 0.954 | d128_scale10/causal/matched |
| (f) nofold a8ee6dc vs orig | accept | 11796480 | 0 | 0 | 0.0119 | b1n64k_32_4/s0 |
| (f) nofold a8ee6dc vs orig | stress | 1703936 | 86573 | 0 | 0.772 | scale10/s1 |


**Reading.**
- **Calibration fails at c = 1.** The original kernel with only its saved LSE moved by one ulp needs c up to 88.5 on
  the cases and stress, and 1.5 / 3.2 on the acceptance shapes:
  - +1 ulp: 427 rows on the cases, 40 on acceptance, 2,349 on stress;
  - -1 ulp: 550 rows on the cases, 5 on acceptance, 2,025 on stress.
  So "round-off of computing the row's sum" (c = 1) is not a bound that the controls pass. A 1-ulp LSE change is input
  round-off, not summation round-off: it scales every P^ of the row by about (1 + ulp(lse)), so |sum t| moves by about
  ulp(lse) * S_r. With ulp(lse) ~ |lse| 2^-23, the expected c is about |lse| / k_r. This is large on short rows of the
  scaled inputs (the worst cases are d256_scale10 and scale10). That is my explanation of the pattern; I have not
  verified it row by row, because LSE is not in the captures.
- **Every candidate passes v5-abs at c = 1** on every suite and all 4 routes, with no exemptions:

  | candidate | max c on the cases | max c on stress | max c on acceptance |
  |---|---|---|---|
  | HEAD vs orig | 0.954 | 0.616 | 0.012 |
  | C3g vs its predecessor | 0.954 | 0.598 | 0.011 |
  | EXPBX64 vs its predecessor | 0.334 | 0.616 | 0.012 |
  | nofold vs orig | 0.954 | 0.772 | 0.012 |

  The rows that failed the ratio form (191 for HEAD on the cases) all move |sum t| by less than one k 2^-23 S_r. They
  fail the ratio form only because their den = sum |t| is a tiny fraction of S_r (cancelling rows).
- **Consequence for the verdicts in section "Verdict".** Under v5-abs with c = 1, forward C3g, EXPBX64 and C4 each pass,
  and so do the cumulative HEAD and the nofold fallback. Under the ratio form v5 as written, C3g and EXPBX64 fail, but
  so does the original against its own 1-ulp LSE controls. Neither form passes both calibration pairs at c = 1.
  - The smallest c that passes the calibration is about 89 on the cases and stress, and 3.2 on acceptance. That is two
    orders of magnitude above every candidate's need.
  - A bound calibrated this way, e.g. an LSE-ulp term |lse| 2^-23 S_r added per row (LSE is not captured yet), would
    pass the controls with all candidates far inside it.
  - This is a rule decision for Hao; nothing here relaxes v5.

## (f) Error clause (TASK.md): "per case at most 1e-5 relative above the original, and the mean over cases does not rise", no stale / full allowance

**Data.** These are production-path `benchmarks/numerics.py capture` runs (relative-L2 error of O / dQ / dK / dV vs the
fp32 reference, TF32 off) on all 72 cases x 4 routes (jobs 21204235-7). Captures are in `data/<tag>/err_<route>.json`.
- The original and HEAD use the frozen production libraries (`base` = 3d4cd5b, `r19_fwd` = HEAD csrc).
- The other builds use the production op path of their v5 DBG libraries. Those run the same kernels with null DBG
  pointers. The original and HEAD DBG libraries were shown byte-equal to production on all 72 cases, and HEAD's frozen
  production vs the c4 DBG library shows 0 changed errors.

Input and reference identities are equal across all captures. The 4 routes give identical errors for every build.

**Tool.** `err_audit.py <cand> <ref>`. Reports are in `~/workspace/haok1402/kda-results/r21-v5/err/<cand>_vs_<ref>.json`.
- Reading A: err_cand <= err_ref + 1e-5 (absolute).
- Reading B: err_cand <= err_ref * (1 + 1e-5) (relative).
- Mean: per output over the cases where both errors are finite.

The original's known non-finite cases are left out of the comparison. In the original, d128_zeros O and stale dK are
NaN; C3g repairs these to 0.0. The d256_zeros NaN cases are unchanged in every build.

Counts are case x output pairs failing; they are the same on all 4 routes:

| build | ref | changed | fail A (matched / stale+full) | fail B (matched / stale+full) | mean rise O | dQ | dK | dV |
|---|---|---|---|---|---|---|---|---|
| HEAD (r19_fwd) | original | 110 | 12 (0 / 12) | 33 (6 / 27) | +3.18e-7 | +3.61e-7 | +1.04e-6 | -3.0e-8 |
| C3g alone | pre-C3g 9be3622 (= original, bitwise) | 107 | 12 (0 / 12) | 32 (5 / 27) | +3.18e-7 | -9.4e-9 | +6.56e-7 | +2.5e-8 |
| EXPBX64 alone | pre-EXPBX64 19c238b | 67 | 2 (0 / 2) | 3 (1 / 2) | 0 | +3.70e-7 | +3.75e-7 | -5.5e-8 |
| EXPBX64 cumulative | original | 110 | 12 | 33 | same as HEAD | | | |
| C4 alone | 7a06674 | **0** (bitwise) | 0 | 0 | 0 | 0 | 0 | 0 |
| C4 cumulative | original | 110 | 12 | 33 | same as HEAD | | | |
| nofold (a8ee6dc) | original | 112 | 11 (0 / 11) | 31 (6 / 25) | +3.17e-7 | -7.3e-7 | +1.72e-7 | -9.6e-8 |
| C3n2 (f96a26b) | original | 105 | 9 (0 / 9) | 30 (5 / 25) | +3.17e-7 | -1.08e-6 | -2.33e-7 | -5.7e-8 |

Worst cases (error original -> candidate):
- **HEAD, reading A**, all stale (stale delta rowsum(dO * O) amplifies the O change):
  - d128_mqa/causal/stale dK 6.5584e-2 -> 6.5794e-2 (+2.11e-4) and dQ +1.49e-4;
  - d128_scale10/causal/stale dK 1.1328 -> 1.1329 (+1.11e-4) and dQ +8.4e-5;
  - d128_gqa3_promote/causal/stale dQ +4.5e-5; d128_scale3/causal/stale dQ +4.0e-5.
- **HEAD, reading B matched failures**, all O from the forward:
  - d128_outliers O +1.04e-4 rel (4.3878e-2 -> 4.3882e-2);
  - d128_gqa4 O +3.8e-5; d128_scale10 O +2.5e-5; d128_scale3 O +1.6e-5; d128_mqa O +1.0e-5;
  - d128_bhnd dK +1.1e-5 (from EXPBX64).
- **C3g alone:** the same stale cases (d128_mqa dK +2.10e-4, ...), plus d128_outliers/full/stale dK +3.5e-5 and O
  +1.45e-4 rel.
- **EXPBX64 alone:**
  - reading A: d128_scale3/causal/stale dK +2.66e-5 and dQ +2.63e-5;
  - reading B: those two plus d128_bhnd/causal/matched dK +1.18e-5 rel.

The changes go both ways. Up vs down among changed case x output pairs: HEAD 51 up / 59 down, C3g 55 / 52,
EXPBX64 28 / 39, nofold 47 / 65, C3n2 47 / 58.

**Verdict under the literal clause, with no allowance:**
- **Forward C3g FAILS.** Reading A: 12 pairs, all stale / full. Reading B: 32 pairs, 5 of them matched O. Mean rises:
  O +3.2e-7, dK +6.6e-7, dV +2.5e-8.
- **EXPBX64 FAILS.**
  - Reading A: 2 pairs, d128_scale3 stale dQ / dK.
  - Reading B: 3 pairs, 1 of them matched (d128_bhnd dK).
  - Mean rises: dQ +3.7e-7 and dK +3.7e-7.
- **C4 PASSES:** it is bitwise to its predecessor on all 72 x 4.
- **HEAD (cumulative) FAILS:** 12 (A) / 33 (B) pairs; means rise on O, dQ and dK.
- **The fallbacks also fail.**
  - nofold: 11 (A) / 31 (B) pairs; its means rise on O and dK.
  - C3n2: 9 (A) / 30 (B) pairs; only its O mean rises.
  - The matched O failures are the same in C3g, nofold and C3n2. So they come from BM192 + FWDLAZY, not from the S-descale
    fold (FWDDFOLD).

**On matched cases only (the tuned path), under reading A every build passes.** The reading-A failures are all
stale / full cases, which v4 allowed up to 1e-3. The worst is +2.1e-4, caused by the stale delta's amplification of
O round-off. Under reading B, the matched failures are O changes of +1.0e-5 to +1.04e-4 relative from the forward
(C3g, nofold, C3n2 alike), and one dK change from EXPBX64. The mean rises are all at most 1.04e-6 absolute; v4's noise
allowance of 1e-6 would have absorbed all but HEAD's dK. "Does not rise" taken literally fails for every non-bitwise
change.
