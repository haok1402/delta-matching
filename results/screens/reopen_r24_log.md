# S24 reopen: output-changing levers re-screened on r24_stack under residual-v5 + the relative error clause

Base: kda-sm90 HEAD 6f4d314 (= r24_stack; frozen `.local/frozen/r24_stack`). Trees `.local/wt/S24-NN` (git archive HEAD +
variant), DBG twins `.local/wt/S24-NN-dbg` (DM_DBG=1 + `apply_v5_dbg.py`), libraries `.local/jit/S24-NN{,-dbg}`.
Patches (vs HEAD) `.local/screens/S24-NN/variant.patch`. Tools `.local/rounds/r24-reopen/` (copies of benchmarks/v5
numerics_v5.py / err_audit.py / apply_v5_dbg.py at HEAD; `screen.sh`, `gate.sh`, `err_audit_r.py` = err_audit on a route
subset + worst matched / stale-full and mean in %, `longerr.py`, `abpair.sh`). Reports
`~/workspace/haok1402/kda-results/r24-reopen/S24-NN/`. Job logs `.local/logs/kda-S24-*-<id>.log` (ids in `jobs.txt`).

References: v5 rows vs the ORIGINAL's DBG captures (`r21-v5/data/orig`; accept regenerated from `.local/wt/v5-orig`,
job 21211915, deleted after use), `numerics_v5.py check --form v5-abs` (c = 1). Error clause vs the ORIGINAL's production
captures (`r21-v5/data/base`) with the round-24 relative clause (matched <= 1.01 x, stale / full <= 1.10 x, per-output
mean <= 1.01 x). r24_stack itself vs the original: worst matched +0.0104% (outliers O), worst stale +0.322% (mqa dK),
max v5 c 0.954; most of the margin below is that inherited amount.
Timing: 2-rep screens (B2 N8192 H16/4, H32/4, causal matched) vs r24_stack, then 3-rep `ab.py --grid paper` pairs.
Screen noise is about +-1% (an untouched kernel moved up to -9.8% in one screen, S24-05 pass 0).

## Per lever

| tag | lever | v5-abs c=1 (cases default) | rel clause: worst matched / worst stale-full / means (O dQ dK dV) | screen (target kernel H16 / H32) | verdict |
|---|---|---|---|---|---|
| S24-01 | pass 0 DEFER (e3275a3): per-lane fp64 row partials, one quad reduce at the end | PASS (c 0.954; 0 abs-worse) | +0.010% / +0.32% / +0.0004 +0.0001 +0.0004 -0.0000 % | pass 0 -3.77 / -3.26% | passes; stacked into 02 |
| S24-03 | pass 0 LAZY alone (dP descale once per 8-key half partial) | PASS (c 0.954) | same as r24 (+0.010 / +0.32%) | pass 0 -4.21 / -4.40% | passes; stacked into 02 |
| S24-02 | pass 0 DEFER + LAZY (e3275a3 + c49c0d6) | PASS on all suites (below) | +0.010% / +0.32% / dK mean +0.0004% | pass 0 -7.40 / -7.85% | **CANDIDATE (pass 0)** |
| S24-04 | pass 0 FOLD residue: folded bias fma(s, dsc, -b) also on the masked (diagonal) tiles, pass 0 only | **FAIL** (13 rows abs-worse in 7 cases, c 11.1) | pass | pass 0 +0.66 / +1.71% | reject (v5 + no time) |
| S24-05 | pass 1 PW16 (dV promote window 8 -> 16 m-tiles) | PASS (rows = r24's: dV only) | +0.034% (bhnd dV) / +0.32% / dV mean -0.008% | pass 1 -2.52 / -1.70% | passes; superseded by PW32 / PW64 |
| S24-06 | pass 1 PW32 | PASS on all suites | +0.066% (gqa3_promote dV) / +0.32% / dV mean -0.004% | pass 1 -3.57 / -3.40% | **CANDIDATE (pass 1)**, at the 3% bar |
| S24-13 | pass 1 PW64 (extension) | PASS on all suites | +0.066% / +0.32% / dV mean -0.003% | pass 1 -4.10 / -4.35% | **CANDIDATE (pass 1, alternative)**, see long-N dV |
| S24-14 | pass 1 PW128 (extension) | PASS (cases default) | as PW64 (cases are <= 48 m-tiles) | pass 1 -4.93 / -3.97% | not pursued: long-N dV +0.57% and rising with the window |
| S24-07 | pass 1 KFOLDSCALE (S23-pr-06 + explicit `__fmaf_rn` in the dS line) | PASS (c 0.954) | +0.010% / +0.32% / dQ +0.0002% dK +0.0004% | pass 1 -0.41 / +1.03%; one more Fp8R instance at STACK 24 | reject: no time (DBG == production now: 0 differing) |
| S24-08 | pass 1 DKACC (S18-nm-12 port: dK in the wgmma accumulator, running basis) | PASS (c 0.954) | **+0.944%** (outliers matched dQ) / +0.82% / dQ +0.017% dK +0.020% | pass 1 -0.34 / -1.13% | reject: < 1% time; long-N dK +1.12% (> 1%) |
| S24-11 | PW32 + DKACC | PASS | +0.944% / +0.82% | pass 1 -4.10 / -3.65%; paper -3.92% (= PW32 alone) | reject: DKACC adds nothing |
| S24-09 | fwd per-group promote (S23-fr-05 idea on BM192: O += D_g w_g, no rho_1) | PASS | **FAIL**: scale3 matched O +3.35%; non-finite 13 vs 6 | fwd +2.03 / +1.17% | bug: the slab path promoted group 0 before the lazy rebase (fixed in 15 / 16) |
| S24-15 | 09 + lazy rebase before the group-0 promote in the slab path | PASS | +0.019% / +0.38% / O +0.0013% dQ +0.0017% dK +0.0027% | fwd +7.34 / +6.78% (forward STACK 24 -> 112) | reject: slower |
| S24-16 | PGP on the steady-state PRETURN tiles only (prologue / epilogue keep the rho chain) | PASS | +0.016% / +0.33% / O +0.0005% dQ +0.0019% dK +0.0028% | fwd +1.33 / +1.68% | reject: slower (one 64-FMUL rho pass becomes 64 FFMA + a second zero-init temp; no gain on 128-key tiles) |

Notes:
- FOLD: the old -13% (on the pre-EXPBX64 stack) is already in r24_stack. EXPBX64 (KDA_PK_EXPB 6) computes exactly the
  folded argument fma(s, dsc, -b), b = fma(lse, log2e, -8), on every unmasked tile in pass 0 and pass 1. The only
  remainder is the masked diagonal tiles (S24-04). It has no time, and v5 rejects it: short causal rows become worse
  than the original, the same as forward S23-fr-03 / 04.
- LAZY's and DEFER's earlier failures were per-row ratio regressions (residual-v4). Under v5-abs every row passes, with
  max c 0.954 on the cases (r24's own value), 0.62 on stress and 0.47 on accept. The ratio-worse rows (410 / 142,886 /
  35) are cancelling rows inside their round-off bound. Gradient errors move by <= 4e-7 relative.
- PW changes only dV, so the v5 rows are the r24 rows (P^, dP and delta are bitwise). The 72 cases stop at N 3072 (48
  m-tiles), so they cannot tell any window >= 64 apart. Supplementary long-sequence dV error vs fp32 (`longerr.py`,
  job 21212210; rel-L2, causal matched; original -> candidate):

  | case | original dV | r24 | PW32 | PW64 | PW128 |
  |---|---|---|---|---|---|
  | B1 H4/1 N16384 gauss | 0.0522697 | 0.0522682 | 0.0522818 (+0.023%) | 0.0523627 (+0.178%) | 0.0525687 (+0.572%) |
  | B1 H4/1 N16384 scale3 | 0.1377337 | 0.1377347 | 0.1377131 (-0.015%) | 0.1376857 (-0.035%) | 0.1376378 (-0.070%) |
  | B1 H2/1 N32768 gauss | 0.0515523 | 0.0515521 | 0.0515244 (-0.054%) | 0.0515995 (+0.092%) | 0.0516817 (+0.251%) |

  PW32 is neutral at long N. PW64 stays within the matched 1% bound but is systematically above the original on
  Gaussian inputs. PW128 grows with the window (wgmma accumulation over more tiles before the fp32 promote).
  Same run, DKACC (S24-11): dK +1.12% and dQ +0.50% at N32768 (dQ +0.25%, dK +0.33% at N16384). So DKACC fails the
  matched 1% clause outside the 72 cases, and its 72-case margin was only 0.056% to begin with. S24-02 at long N is
  within 5e-7 relative of r24 on every output.

## Full gates (all 4 routes, stress, accept 3 seeds; vs the ORIGINAL)

| tag | err clause, 4 routes (identical) | v5-abs cases x 4 | stress (1,703,936 rows) | accept (11,796,480 rows) | DBG == prod | pytest | jobs |
|---|---|---|---|---|---|---|---|
| S24-02 | 0 fail; worst matched +0.0104%, stale +0.32%; means <= +0.0004% | PASS, c 0.954 | PASS, c 0.622 | PASS, c 0.471 | yes (0 differ) | 49 passed | 21212207, 21212873 |
| S24-06 | 0 fail; worst matched +0.0655%, stale +0.32%; dV mean -0.004% | PASS, c 0.954 | PASS, c 0.616 | PASS, c 0.012 | yes | 49 passed | 21212208, 21212873 |
| S24-13 | 0 fail; same as 06 on the cases | PASS, c 0.954 | PASS, c 0.616 | PASS, c 0.012 | yes | 49 passed | 21212871, 21212873 |
| S24-17 (02 + 06) | 0 fail; worst matched +0.0655%, stale +0.32% | PASS, c 0.954 | PASS, c 0.622 | PASS, c 0.471 | yes | 49 passed | 21212624, 21213012 |

No ptxas spill or serialization diagnostics in any build. Register / stack usage of S24-02 / 06 / 13 / 17 equals
r24_stack's.

## Timing pairs (3-rep `ab.py --grid paper` vs r24_stack; target over all D128 shapes; checks target_gain / tune_shapes / complete_set all True)

| tag | target | job 1 | job 2 | paper-set rows (job 1 / job 2) |
|---|---|---|---|---|
| S24-02 | pass 0 | **-7.95%** (21212323) | **-8.08%** (21212625) | pass 0 -8.29 / -8.46%; e2e fwd+bwd -1.07 / -1.06%; others within +-0.5% |
| S24-06 PW32 | pass 1 | **-3.74%** (21212323) | **-3.01%** (21212625) | pass 1 -4.07 / -3.58%; e2e -2.28 / -1.48% |
| S24-13 PW64 | pass 1 | **-4.01%** (21212323) | **-4.50%** (21212625) | pass 1 -3.76 / -3.90%; e2e -1.88 / -1.73% |
| S24-11 PW32 + DKACC | pass 1 | -3.92% (21212323) | - | rejected on numerics (long-N dK) |
| S24-17 02 + 06 | pass 1 (pass 0 + pass 1 summed) | pass 1 -3.30%; pass 0 -8.02%; pass 0 + pass 1 -4.23%; e2e -2.71% (21212874) | pass 1 -3.22%; pass 0 -8.05%; pass 0 + pass 1 -4.24%; e2e -2.40% (21212881) | pass 2 +0.33 / +0.38% (untouched) |

bench.py was not run (left to the integrator's gate).

## Verdict

- **Pass 0: S24-02 (P0DEFER + P0LAZY), CANDIDATE.** Pass 0 -7.95% / -8.08% in two jobs. v5-abs passes on every suite
  and route, the relative error clause passes with the means unchanged at 1e-6 relative, DBG == production, and pytest
  passes. Tree `.local/wt/S24-02`.
- **Pass 1: S24-06 (PW32), CANDIDATE at the bar.** -3.74% / -3.01% in two jobs; it passes every numerics check and is
  neutral on long-N dV. The stronger alternative is **S24-13 (PW64)**: -4.01% / -4.50%, same gate results, but long-N dV
  +0.18% / +0.09% vs the original (inside 1%, and invisible to the 72 cases). That choice is for the integrator.
  PW16, PW128, KFOLDSCALE and DKACC are rejected as above.
- **Stack: S24-17 = S24-02 + S24-06.** It passes the full gate. Pass 0 + pass 1 summed -4.23% / -4.24%, e2e fwd+bwd
  -2.71% / -2.40%.
- **Forward: per-group promote on BM192, REJECTED.** With the rebase fix it passes numerics but is 1.3-1.7% slower
  (S24-16), or 7% slower with the epilogue form (S24-15).
- **FOLD: nothing left on r24_stack.** EXPBX64 already holds it; the masked-tile remainder fails v5.
