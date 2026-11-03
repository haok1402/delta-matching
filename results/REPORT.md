# kda-sm90: delta-matching fp8 attention on H100 — interim report

Status: interim (round 24; the round-23 convergence was under the absolute error clause, superseded by Hao's relative
clause). Branch `kda-sm90`; accepted build `r24_stack` (kda-sm90 6f4d314 + tooling). Latest status in the last section. Every number below is traced to a committed file; per-round detail and every screened variant are in
`results/rounds.md`, `results/screens/` and the helpers' `CANDIDATES.md` branches; rules in
`docs/kda-sm90/rule-change-r10.md`.
JSON results (raw A/B runs, medians, calibrations, sol4 / numerics reports) live outside the repo in
`~/workspace/haok1402/kda-results/` under the same relative paths as the `results/...` paths cited here (round 20);
only `results/progress/A_median.json` / `B_median.json` stay in git.

## Scope and goal

- Method: the matched-delta (Delta-Matching) fp8 attention, D128, causal.
- Acceptance shapes (Hao, round 16): the training shapes of the paper's 1.67B GDN/GQA hybrid (Hq 16 / Hk 4) and
  Qwen3-30B-A3B attention (Hq 32 / Hk 4) at B2 N8192 and B1 N65536; equal weight (geometric mean of per-shape ratios).
- Acceptance: target kernel >= 3% faster in two fresh jobs, no shape slower beyond spread, bench within 1%; bitwise on
  72 cases x 4 routes, or (named numerics changes) the residual-v4 numerics contract on all four routes.
- Goal (Hao, round 17): forward, pass 0, pass 1, pass 2 each >= 90% of the corrected floor sol4c (below), on the
  fixed-reference calibration; first milestone: forward as fast as FA3's fp8 forward.

## Before / after (four acceptance shapes, summed medians, ms; original = main 3d4cd5b, now = r17_pk)

Source: `results/progress/ab.md` (job 21181967, 3 reps each, same job).

| kernel | original ms | now ms | change |
|---|---:|---:|---:|
| forward | 75.302 | 67.493 | -10.4% |
| pass 0 (matched delta) | 63.185 | 52.880 | -16.3% |
| pass 1 (dK / dV, dS') | 198.397 | 180.046 | -9.3% |
| pass 2 (dQ) | 41.889 | 37.873 | -9.6% |
| glue (copies / GQA fold) | 5.010 | 3.181 | -36.5% |
| producers fwd / bwd, twin | 2.452 | 2.489 | +1.5% |
| end-to-end forward | 77.476 | 67.362 | -13.1% |
| end-to-end fwd+bwd | 385.356 | 342.276 | -11.2% |

## Floors: sol4-v2 and sol4c (accepted build r17_pk, fixed-reference calibration 1481.2 TFLOP/s fp8, 3026 GB/s)

Source: `results/r17/accepted_pk/sol4c.md` (`benchmarks/sol4.py`).

| kernel | measured ms | sol4-v2 floor | % of sol4-v2 | sol4c floor | % of sol4c | target 90% |
|---|---:|---:|---:|---:|---:|---|
| pass 1 | 179.293 | 73.665 (tensor) | 41.1% | 118.815 | 66.3% | not met |
| forward | 67.089 | 36.815 (tensor) | 54.9% | 52.668 | 78.5% | not met |
| pass 0 | 52.715 | 36.920 (tensor) | 70.0% | 49.557 | 94.0% | met |
| pass 2 | 37.807 | 35.592 (bytes) | 94.1% | 35.592 | 94.1% | met |

- sol4-v2: per kernel the max of four floors: tensor (tile-granular FLOPs / calibrated fp8 GEMM peak), MUFU and issue
  (loop-only lower bounds from a control-flow ledger of the emitted step loops), bytes (route-aware compulsory traffic /
  calibrated copy bandwidth). Inputs are bound (calibration digest, library hash, exact grid).
- sol4c (Hao, after kda-solcheck): sol4-v2 / the fraction an IDEAL kernel with exactly the accepted step-loop
  instruction mix reaches on this H100 (forward 0.699, pass 1 0.620, pass 0 0.745; best of five schedules; tensor and
  ALU work couple at ~C_alu + 0.55 C_tensor; loaded clock 1.66-1.80 GHz; IPC <= ~3.5; origin/kda-sm90-solcheck
  results/solcheck/README.md). 90% of sol4-v2 itself is not reachable with these instruction mixes.
- The same-job fp8 GEMM calibration varies ~5% between jobs; goal attainment uses the highest observed peak (1481.2).

## FA3 and paper comparisons

- Forward kernel vs FA3's fp8 forward kernel on the four shapes: 1.155x (r17_pk; was 1.32x before round 17).
- Paper protocol (Appendix C.2; `results/paper_c2.md`, job 21181967), B2 N8192, TFLOP/s fwd / bwd / fwd+bwd:

| config | accepted | original | FA3 BF16 | cuDNN BF16 | paper (matched) |
|---|---|---|---|---|---|
| 16 / 4 | 582.8 / 486.4 / 510.5 | 516.4 / 442.3 / 461.0 | 697.0 / 499.7 / 543.6 | 590.3 / 483.5 / 509.6 | 661.0 / 421.5 / 470.2 |
| 32 / 4 | 639.0 / 500.3 / 533.3 | 564.4 / 452.6 / 479.6 | 723.1 / 519.0 / 564.5 | (see paper_c2.md) | - |

  The protocol's per-call forward window includes ~0.3 ms of idle-GPU launch latency (host work in the op is 0.015 ms);
  the forward kernel itself ran at 704 TFLOP/s already at r14_h3. The paper's FA3 BF16 numbers (260.7 fwd+bwd at 16/4)
  do not reproduce on this H100 (549 here), so its "1.80x FA3 BF16" ratio does not either; cuDNN BF16 matches the paper.

## Accepted changes (in order; details and evidence in results/rounds.md)

| round | change | kernel | gain (gate / confirmation) | class |
|---|---|---|---|---|
| 3 | DO2 (dO pipeline depth 2) | pass 1 | (historical grid) | bitwise |
| 9 | P2ST3: pass-2 hd128 pipeline depth 3 | pass 2 | -7.5% / -6.6% | bitwise |
| 10 | PSIPF: psi loaded one iteration early | pass 2 | -3.21% / -3.14% | bitwise |
| 12 | H1a + H1b stack: dV before the psi barrier, dS' words pre-packed | pass 1 | -4.03% / -3.99% | bitwise |
| 14 | H3: producers read Q / K / V through their own strides | glue | -67.75% / -67.75% | bitwise |
| 17 | C2: P0RTR transposed quad reduce + P0SPF 8-byte scale loads (kda-pass0) | pass 0 | -5.21% / -5.27% | bitwise |
| 17 | C3: P0LPT longest-first dispatch + P0W4 four warpgroups per CTA (kda-pass0) | pass 0 | -7.67% / -7.64% | bitwise |
| 17 | S17-p1-39: early dV rescale + deferred double-buffered dS' stores | pass 1 | -4.68% / -4.87% | bitwise |
| 17 | C3g: BM192 three-warpgroup forward + lazy O rescale + descale fold on unmasked steps (kda-fwd) | forward | -12.89% / -12.68% | named numerics |
| 17 | pass-0 stack: P0NML mask-free loop + register split + P0ALL one all-heads call (kda-pass0) | pass 0 | -3.19% / -3.43% | bitwise |
| 17 | EXPBX64: exp-line bias fold off the masked diagonal, passes 0 + 1 (kda-p1packed) | pass 0 | -4.50% / -4.04% | named numerics |
| 18 | C4: SHFL8 dS' transpose + exp clamp on the masked-tile line only (kda-pass1) | pass 0 + pass 1 | p0 -3.53 / -3.86%, p1 -3.31 / -3.08% | named numerics (bitwise on all cases) |
| 19 | forward stack: PRETURN + k/v-scale smem staging + guard / scale-read / weight / barrier-id moves (kda-fwd, kda-fwdport, S19 worker) | forward | -3.51% / -3.92% | bitwise |

## What was tried and why it failed (summary; full ledgers in results/screens and the helpers' CANDIDATES.md)

- Forward (~90 variants incl. kda-fwd S1-S54): every bitwise variant other than the accepted ones within noise or
  slower; probes put the rho rescales at ~9% and the S descale at ~5% of the forward; shared-basis P codes (no rho)
  -4.4% but O error +1.3e-3..4.2e-3 (fails residual-v4 by 100-400x: real accuracy loss); scheduler-barrier alternatives
  +1..12%; QK splits and in-step branches spill at 160 registers; held bitwise stack PRETURN + k/v-scale staging -2.4%.
- Pass 1 (~180 variants incl. kda-pass1's ~135): PINGPONG (round 6), MUFU-poly v1 (+5.6%), per-warpgroup psi
  (pass 2's second accumulator +7..9%), freeing registers for a pipelined S / dP (kda-p1split, +3.5-4%); per element
  8 instructions remain, each fixed by the numerics contract, without an sm90 shortcut, neutral, or failing
  residual-v4 (EXPB in pass 1, PW16 / PW32, windowed dK); held SHFL8 -1.1% (bitwise).
- Pass 0: output-changing cuts (FOLD -13%, LAZY, DEFER, NORM) fail residual-v4 per-row non-regression; bitwise levers
  exhausted after the accepted ones; meets sol4c.
- Pass 2: at ~94% of its bytes floor since round 10.
- Tooling / rules: four-floor model sol4-v1 -> v2 (control-flow ledger, bound inputs, route-aware glue), the numerics
  contract residual-v2 -> v3 -> v4 (per-row non-regression at round-off plus the measured LSE +-1 ulp envelope,
  degenerate / ill-conditioned exemptions fixed by the accepted build, error tolerances 1e-5 matched / 1e-3 stale-full,
  mean noise 1e-6, inherited-defect repair), equivalence manifest v4.

## Open (round 18 onward, Hao's direction)

1. Structural pass-1 redesign reusing pass 0's work (pass 0 stores what pass 1 needs, or a two-phase pass 1 that
   computes the matched delta itself), from kda-fuse's measured structure.
2. Cheaper forms of the exp clamp and the fp32 dK fold as named numerics changes under residual-v4.
3. Completing the held stacks (forward -2.4%, pass 1 -1.1%) to >= 3%.
4. FA3 fp8 forward parity (now 1.155x).

## Update: round 18 (accepted build r18_c4, job 21191934)

| kernel | sol4-v2 % (fixed ref.) | sol4c % (fixed ref.) | sol4c % (same-job 1454.9 TFLOP/s) | target 90% |
|---|---:|---:|---:|---|
| pass 1 | 42.6% | 68.8% | 70.0% | not met |
| forward | 55.0% | 78.6% | 80.1% | not met |
| pass 0 | 73.3% | 98.4% | 100.2% | met |
| pass 2 | 94.3% | 94.3% | 94.3% | met |

- Original -> now (four shapes, same job): e2e forward -14.0%, e2e fwd+bwd -13.1% (`results/progress/ab.md`).
- Forward / FA3 fp8 forward: 1.162x (calibration and clock spread between jobs ~1%).
- Paper protocol 16/4 (B2 N8192): 568.9 / 505.3 / 521.9 TFLOP/s fwd / bwd / fwd+bwd (cuDNN BF16 582.7 / 486.2 / 510.2;
  FA3 BF16 691.0 / 498.9 / 541.9); 32/4: 628.0 / 522.3 / 548.6.
- Round-18 work without a candidate: structural pass 0 / pass 1 reuse (pass 1 computing the delta needs a cross-CTA
  exchange costing ~+100% on pass 1 and cannot run at 64K; handing pass 0's products to pass 1 costs more bytes than
  the backward); cheaper dK fold (neutral or fails residual-v4); forward shared-basis P codes (fails residual-v4 by
  100-400x). Every helper (forward, pass 0, pass 1) reports DONE.

## Update: round 19 (accepted build r19_fwd, job 21196592; same-job calibration 1424.2 TFLOP/s)

| kernel | sol4-v2 % (fixed ref. 1481.2) | sol4c % (fixed ref.) | sol4c % (same job) | target 90% |
|---|---:|---:|---:|---|
| pass 1 | 42.9% | 69.2% | 71.9% | not met |
| forward | 57.6% | 82.4% | 85.7% | not met |
| pass 0 | 72.7% | 97.6% | 101.5% | met |
| pass 2 | 94.3% | 94.3% | 94.3% | met |

- Forward / FA3 fp8 forward: 1.108x (was 1.162x).
- Original -> now (four shapes, same job): e2e forward -15.8%, e2e fwd+bwd -13.3%.
- Paper protocol 16/4: 621.5 / 500.3 / 529.7 TFLOP/s fwd / bwd / fwd+bwd (`results/paper_c2/r19_fwd_table.md`).
- Forward SASS diff vs FA3 fp8 (results/screens/fwd_sassdiff.md): 9.2 vs 5.9 instructions per score per thread; the
  ring turn's remaining 178 instructions are 128 rho / promote (contract-fixed), 8 QGMMA and ~42 descriptor / sync.

## Round 20 (no acceptance): the layout spaces are exhausted

- Forward: every layout in the space the numerics and the 65536-register file allow was screened (K/V depth, register
  splits, 2 CTAs / SM, 128-row tiles, producer warp, key-tile widths, 4 warpgroups) plus an FA4-style softmax / PV
  warpgroup split (+38%); the accepted 192 x 128, 3-warpgroup layout is the best point. The group-max shuffle chain
  (~1.9%) and the per-group rho / promote (128 instructions per 128 keys) are the remaining gap to FA3 fp8 (1.108x).
- Pass 1: pipeline depths, a persistent scheduler, L2 tile order and register splits screened; the 64 x 128,
  2-warpgroup geometry is the only workable point under the per-tile scale contract and the 240-register budget.
- Further progress needs a different kernel structure (kda-bwdport: pass 1 rebuilt from FA3's BF16 backward) or a
  change to the numerics contract itself (per-group P basis, per-tile psi), which the current rules do not admit.

## Update: round 21, residual-v5 re-audit and the clean baseline r21_clean (job 21206028, calibration 1436.2 TFLOP/s)

The stricter numerics contract (TASK.md residual-v5: every matched row within the summation round-off of its residual
in the ORIGINAL kernel, no exemptions; per-case fp32 error at most 1e-5 above the original and no rise of the mean)
was applied to every accepted numerics change (`results/residual_v5_audit.md`). All passed the row check; the error
clause failed for forward C3g (its 192 x 128 tiling: 32 of 33 failing case-outputs) and EXPBX64 (1). Both were
REVERTED; C4 (bitwise) and every bitwise change stay. The clean build is byte-identical to the original on all 72 cases
x 4 routes except the repaired all-zero-input defect.

| kernel | measured ms | sol4-v2 % (fixed ref.) | sol4c % (fixed ref.) | sol4c % (same job) | target 90% |
|---|---:|---:|---:|---:|---|
| pass 1 | 176.506 | 41.7% | 67.3% | 69.4% | not met |
| forward | 73.846 | 50.0% | 71.5% | 73.8% | not met |
| pass 0 | 54.355 | 67.9% | 91.2% | 94.0% | met |
| pass 2 | 37.692 | 94.5% | 94.5% | 94.5% | met |

- Forward / FA3 fp8 forward: 1.285x (r19_fwd was 1.108x with C3g).
- Original -> now (four shapes, same job): e2e forward -4.7%, e2e fwd+bwd -8.3% (`results/progress/ab.md`); r19_fwd's
  -13.3% overstated the speedup that survives the stricter numerics.
- Paper protocol 16/4: 534.7 / 490.6 / 502.4 TFLOP/s fwd / bwd / fwd+bwd (`results/paper_c2/r21_clean_table.md`).
- The accepted-changes table above keeps C3g and EXPBX64 for the record; both are reverted as of round 21.

## Final: convergence under the TASK.md contract (round 23)

Accepted build r21_clean (kda-sm90 e80de8f): the ORIGINAL arithmetic in every kernel (byte-identical to main 3d4cd5b on
all 72 cases x 4 routes except the repaired all-zero-input NaN defect), plus every bitwise speedup accepted in the
campaign. On the four acceptance shapes vs the original (same job, 21206028): e2e forward -4.7%, e2e fwd+bwd -8.3%;
pass 0 -12.6%, pass 1 -9.6% (revert-tree timing), glue -36%, pass 2 -10%.

| kernel | % of sol4c (fixed ref.) | target 90% | status |
|---|---:|---|---|
| pass 0 | 91.2% | met | converged (bitwise levers exhausted; numerics cuts fail v5 / the error clause) |
| pass 2 | 94.5% | met | at its bytes floor |
| forward | 71.5% | not met | converged under the contract (1.285x FA3 fp8) |
| pass 1 | 67.3% | not met | converged under the contract |

Why the forward and pass 1 stop here (evidence: results/rounds.md rounds 16-23, results/screens/*, helpers' CANDIDATES.md):
- Forward: per 256-key step and warp it executes 1277 instructions vs FA3 fp8's 771; the gap is the faithful
  per-64-key P basis (192 FMUL rho rescales, 128 FMUL descale, 64 FFMA promote, group maxima, rho / weight exps). Every
  layout the 65536-register file allows was screened (the 2-WG x 128 x 256 tile is optimal; BM192 is faster but its
  128-key tiling raises O / stale-mode errors above the original and was reverted); turn scheduling does not set the
  step time; bitwise-removable instructions are ~3% (< 1% time). Accuracy-preserving reformulations (descale folds
  with shared or backward-matched factors, per-group promotes, power-of-two bases): the folds pass the v5 rows but
  move stale-mode gradients by up to ~1e-4 per case and shift the per-output means randomly (a shared per-group
  factor biases every score of a group), so none passes "per case <= +1e-5 and the mean does not rise".
- Pass 1: scheduling, layout (64 x 128 tiles, 2 WGs at 240 registers is the only workable point), persistent
  scheduling, per-warpgroup psi, structural reuse of pass 0's work (cross-CTA exchange ~+100% of pass 1; product
  handoff costs more bytes than the backward) and in-kernel dQ were measured slower or neutral; the per-element path
  (exp 3, dS 2, amax 1, scale 1, e4m3 1) has no cheaper exact form on sm90; power-of-two psi is provably coarser in the
  subnormal band (52 failing case-outputs); the one strictly-better-bound form (KFOLDSCALE) raises the dK mean by
  5e-9 and gains no time.
- What would move them: a contract change only Hao can make: a mean clause that tolerates round-off-level noise
  (e.g. the lazy rescale e256nf: per case <= 5.7e-6, means < 1e-7, -1%; the backward-matched descale fold -3..-4%), or a
  shared per-tile P basis / psi convention (FWDSHB -4.4% forward but O error +1e-3: a real accuracy loss, not advised).

Campaign totals: 11 changes accepted over rounds 3-19 (two later reverted under residual-v5), several hundred variants
screened and logged (results/screens/, helpers' CANDIDATES.md), the four-floor model sol4-v2 / sol4c, the numerics
contract evolved from bitwise-only to residual-v5 with the original as the fixed reference.

## Update: round 24, relative error clause; C3g + EXPBX64 re-accepted (r24_stack, job 21211208, calibration 1371.6 TFLOP/s)

Hao kept residual-v5's zero-sum rows as the hard requirement and replaced the absolute error clause by a relative one
(per case <= 1.01 x the original's error in matched mode, <= 1.10 x in stale / full; mean <= 1.01 x). Under it the
reverted changes pass (C3g worst +0.32% of the original's error, EXPBX64 +0.011%) and were re-gated stacked on
r21_clean: forward -12.8 / -12.9%, pass 0 -6.6 / -6.5%, e2e fwd+bwd -5.2 / -5.0% in two jobs.

| kernel | measured ms | sol4-v2 % (fixed ref.) | sol4c % (fixed ref.) | target 90% |
|---|---:|---:|---:|---|
| pass 1 | 172.54 | 42.7% | 68.9% | not met |
| forward | 64.05 | 57.5% | 82.2% | not met |
| pass 0 | 50.81 | 72.7% | 97.5% | met |
| pass 2 | 37.82 | 94.4% | 94.4% | met |

- Forward / FA3 fp8 forward: 1.108x.
- Original -> now (four shapes, same job): e2e forward -16.1%, e2e fwd+bwd -13.4% (`results/progress/ab.md`).
- Paper protocol 16/4: 613.5 / 498.1 / 526.3 TFLOP/s fwd / bwd / fwd+bwd (`results/paper_c2/r24_stack_table.md`).
- Reopened levers under the relative clause are being re-screened (round 24 onward).
