# Rule change (Hao, round 10): head dim 128 is the target; one grid

Effective from round 10, replacing AC-3.1 / AC-3.2's affected-subset and held-out wording and the scope of the
per-kernel stop rule:

- One grid: no held-out split. The former main and held-out shapes form one grid (`kernels.py --grid all`), measured
  and tuned together.
- Target: the head-dim-128 kernels (pass 1, forward, pass 2, pass 0, glue at D128), ranked by modeled waste over all
  D128 shapes. Head dim 256 is not tuned and needs no counterpart; it must stay correct (pytest, the four 72-case
  equivalence routes) and must not get slower.
- Acceptance of a candidate targeting kernel K (`ab.py decision()`, enforced by `gate.py --target K`):
  1. K's summed time over all D128 shapes is at least 3% lower than the previous accepted revision;
  2. no D128 shape is slower beyond its run-to-run spread;
  3. no D256 shape is slower beyond its run-to-run spread, and K's summed D256 time is not slower beyond the spread of
     its per-repetition sums;
  4. both the full gate job and a second fresh confirmation job (same candidate and previous-accepted hashes) pass 1-3;
  plus the unchanged requirements: numerics (AC-2), no new spills / stack / outlining / serialization (AC-3.3),
  bench.py at D128 / D256 N8192.
- "Slower beyond its run-to-run spread" (operational definition, recorded for review): the shape's median fwd+bwd
  increase exceeds the larger of its A and B repetition ranges (max - min over the 3 alternating repetitions). The
  stricter reading "every B repetition above every A repetition" flags about 1 in 20 unchanged shapes at 3 x 3
  repetitions (2 false rejections expected per job over 39 D128 shapes).
- Counters: pass 1 and forward reopen at D128 with fresh counters (their closed rounds mixed D256 into the target);
  hypotheses come from the existing D128 profiles. Pass 2 and glue keep their counters. Rounds 3 (DO2, a D256 pass-1
  change) and 9 (P2ST3, a D128 pass-2 change) stay committed.

This file records the user's rule change; docs/kda-sm90/plan.md is frozen while the RLCR loop runs (the loop's
hook forbids editing it), so the plan's AC-3.1 / AC-3.2 text is superseded by this file from round 10 on.

## Direction after the independent audit (Hao, round 11; audit: ~/workspace/haok1402/audits/kda-sm90/REPORT.md)

Extends the D128-only, all-shapes rule above:

1. Stacking: 3% per candidate stays, but a rejected change with identical outputs that gave >= 1% on its target kernel
   (confirmed by two jobs) goes into a stack; stacked changes are combined into one candidate that must clear 3%
   together. First stacked candidate: H1a + H1b (+ AMAX-REDUX if ready) on the current base, gated as one D128 pass-1
   candidate.
2. Build the numerics gate for output-changing candidates (Q2 / Q3: fp32 error per case, determinism, matched-mode
   rowsum residual, dS' / psi capture) now.
3. Pass 1 at D128, in order: per-warpgroup psi with decoupling only; then trace the PINGPONG stage-1 build and redo the
   turn over all four GEMMs; then MUFU-poly exp2. A lever counts as failed only after two designed variants.
4. Between expensive pass-1 rounds, cheap rounds with identical outputs: H3 (remove the Q / K / V .contiguous()
   copies and glue at D128), the forward k-scale prefetch.
5. Speed-of-light model: per kernel the max of the tensor, MUFU, issue (SASS opcode counts) and byte floors, with peaks
   calibrated by microbenchmarks in the same job (cuBLASLt fp8 GEMM, device-to-device copy); FA3 fp8 forward is the
   practical forward reference.
6. Codex review findings block a round only if they could change an accept / reject verdict; everything else is
   queued for the final report. At most one bounded tooling commit per round.
7. gate.py: per-symbol STACK / LOCAL growth is a flagged warning that still runs the A/B; ptxas spill or serialization
   diagnostics stay hard fails.
8. Builds and CPU tests out of GPU jobs where possible; build all variants of a round in one job that times them right
   away; never rebuild inside gate.py when the .so exists.
9. Wait on job IDs or PIDs only, never pgrep / pkill patterns on this shared account.

## Scope change (Hao, round 12): the matched-delta method only

- Target grid: D128, causal, mode "matched" only (matched supports causal only): the 13 D128 shapes of the one grid
  (`kernels.py --grid d128m`). Stale mode and full attention are dropped from ranking, tuning and acceptance; they
  (and D256) must stay correct (tests, the four 72-case equivalence routes) but are not measured for gains or
  regressions.
- Kernels that matter: those a matched forward+backward runs: forward, producers and glue, pass 0 (matched delta),
  pass 1, pass 2; re-ranked by waste over the D128 causal matched shapes. Pass 0 (never profiled) gets a place in the
  order.
- Acceptance as before (>= 3% on the target kernel summed over the tuned configurations, stacking of confirmed >= 1%
  identical-output gains, a second fresh job, no D128 causal matched shape slower beyond its spread), measured on this
  grid only; bench.py predicate on D128 matched only. Enforced by `ab.py decision()` (`is_tuned`) and `gate.py`
  (A/B on `--grid d128m`). The shorter A/B jobs leave room for more candidates.

## Sequence-length focus (Hao, round 13; replaces the round-12 target set)

- Target grid: D128 causal matched at N = 4096 and 8192, in four head / GQA variants (H/Hk 16/16, 16/4, 32/8, 8/8;
  B for ~32k tokens per call: B 8 at 4k, B 4 at 8k): 8 target configurations (`kernels.D128_TARGET`). Kernels are
  ranked and changes accepted (>= 3% on the target kernel summed over these, stacking, a second fresh job) on it.
- No-regression guard: the same variants at N 16384 and 32768 (`kernels.D128_GUARD`): no shape slower beyond its
  run-to-run spread.
- 2k only has to stay correct (tests, equivalence); not measured.
- `kernels.py --grid d128m` = target + guard (16 configurations, causal matched); `ab.py decision()` sums the target
  over the target set and checks shape slowdowns over target + guard.

## Round 16: training-shape focus (Hao, final of four messages during round 16)

- Acceptance set: the models' TRAINING shapes (paper arXiv 2609.37852, B.2.4): D128 causal matched B2 N8192
  (pretraining, per-GPU batch 2 x 8192) and B1 N65536 (the 8K-to-64K context extension), each at Hq 16 / Hk 4 (the
  paper's 1.67B GDN/GQA hybrid) and Hq 32 / Hk 4 (Qwen3-30B-A3B attention): four shapes, equal weight per shape.
- Acceptance: the target kernel >= 3% faster over the set (equal weight: the geometric mean of the per-shape median
  ratios; the plain sum is recorded too), stacking of confirmed >= 1% bitwise gains, a second fresh job, no shape of the
  set slower beyond its spread. 4k / 16k / 32k (evaluation lengths), other head counts, D256, stale and full attention
  only stay correct. The 64K shapes may run with fewer repetitions if a job gets too long, but run in both jobs.
- Report: `results/paper_c2.md`, the paper's Appendix C.2 protocol at B2 N8192 for 16/4 and 32/4, after every accepted
  round (first: the accepted build and the original baseline).
- `kernels.py --grid paper` = the four shapes; `ab.py decision()` gates them (no separate guard); gate.py A/Bs on it.

## Round 16/17: residual contract decided by Hao (AC-2.3) -> "residual-v2"

Hao's decision: per row, the matched-mode residual |sum_j dS_ij| / sum_j |dS_ij| must stay at fp32 round-off (the
existing bound sk * 2^-23 scale), except degenerate rows whose sum_j |dS_ij| is below a tiny floor (one visible key,
all-zero dS), which are exempt; the number of exempt rows is reported per case and the exemption set must not grow vs
the accepted build; no non-exempt row may exceed its own residual in the accepted build by more than round-off
(per-row, not group maxima); the other AC-2.3 conditions stay (deterministic, error vs fp32 not larger on any case,
named numerics change in the commit).

Implementation (benchmarks/numerics.py, version "residual-v2"; the checker derives every bound from the case geometry,
captures never supply tolerances):
- per matched row r (case, backward group, bh, query row) with k_r visible keys (causal: min(N, r + 1)): t_rj are the
  kernel's dS' terms P^_rj (dP_rj - delta_r), reduced from its fp32 partials in fp64 (n-tile ascending, warpgroup);
  num_r = |sum_j t_rj|, den_r = sum_j |t_rj|, ratio_r = num_r / den_r (0 when den_r = 0).
- exempt (degenerate): k_r <= 1, or den_r < 1e-30 (all-zero dS; the kernels' own psi floor).
- predicates: den_r == 0 implies num_r == 0; every non-exempt row ratio_r <= k_r * 2^-23; per-row non-regression
  ratio_r(candidate) <= ratio_r(accepted) + k_r * 2^-23 on non-exempt rows; the candidate's exempt set is a subset of the
  accepted build's (no growth, no relocation); exempt counts reported per case; the candidate's two DBG runs give
  identical row records; recorded visible-key counts equal the geometry.
- the rows (num, den, keys) are stored per capture in a sidecar (`<capture>.rows.pt`) whose sha256 is in the capture;
  `gate.py --mode numerics` accepts only when the check of every one of the four routes reports `ac23_eligible`.
- the plan ratio's group maxima and the centered ratio stay in the report as diagnostics.

## Round 17: the goal and stop rule (Hao, replacing the old stop rule)

Every main kernel of the matched path (forward, pass 0, pass 1, pass 2) must reach at least 90% of its calibrated
four-floor limit (sol4, calibrated on this H100) on the acceptance shapes (B2 N8192 and B1 N65536 at 16/4 and 32/4,
D128, causal, matched). First milestone: the forward at least as fast as FA3's fp8 forward on the same shapes. Two
failed rounds on a lever mean moving to another lever or a structural redesign, not stopping. If a floor looks wrong
(work the method needs that the model does not count), show the evidence and propose a corrected floor instead of
declaring it unreachable. Every accepted round reports percent of the calibrated limit per kernel.

Breadth (Hao, round 17): compute is not rationed: many candidates and many variants per lever, built in parallel
worktrees and screened with short 1-GPU sbatch --wait jobs (`.local/jobs/screen.sh`: build, default-route bitwise
check, 2-rep A/B on B2 N8192 16/4 and 32/4); screen winners go to the full gate + confirmation; every screened variant
is logged with its number (results/screens.md).

## Round 17: residual-v3 (Hao's decision after the accepted build's self-certificate)

The self-certificate of the accepted build (jobs 21171647-50, results/r17/cert) showed the ORIGINAL kernel exceeds the
absolute round-off bound k_r 2^-23 on 10204 of 334592 matched rows on every route (the scale3 / scale10 / outliers
cases and a few 2-3-key rows of the Gaussian cases; a |delta|-conditioned variant still left 9667), so Hao dropped the
absolute bound. An output-changing candidate passes the residual check when:
1. no non-exempt row's residual exceeds its value in the accepted build by more than fp32 round-off:
   ratio_r(candidate) <= ratio_r(accepted) + k_r 2^-23 (per-row non-regression);
2. the degenerate exemption set (k_r <= 1 or sum |t| < 1e-30) does not grow (or move) vs the accepted build;
3. the other AC-2.3 conditions hold: deterministic (identical outputs and row records over two runs), error vs fp32 not
   larger on any case / output, named numerics change in the commit; plus the existing identity / workspace checks.
den_r == 0 still requires num_r == 0. Rows above the absolute bound are reported (`violations`), not enforced.
Implemented as "residual-v3" in benchmarks/numerics.py; `gate.py --mode numerics` accepts only when all four routes'
reports are `ac23_eligible` under residual-v3. The accepted build is re-certified under it before any numerics
candidate is accepted.

## Round 17: residual-v4 (Hao's tolerance decision) and autonomy

residual-v4 (benchmarks/numerics.py, gate --mode numerics), replacing v3:
- Error vs fp32: per case and output tensor, the candidate's relative-L2 error may exceed the accepted build's by at
  most 1e-5 (absolute, in relative-error units: the literal reading of "the relative error may rise by at most 1e-5";
  the per-case check against the frozen baseline uses the same tolerance), and the mean over all cases of each output
  tensor may not rise vs the accepted build.
- Row residual: strict per-row non-regression (accepted value + k_r 2^-23) on well-conditioned rows; rows whose
  residual in the ACCEPTED build is above 1e-3 are ill-conditioned and exempt from it, with their count and worst
  movement reported per case; the ill-conditioned set is fixed by the accepted build (a candidate row that was
  well-conditioned and rises above 1e-3 is a regression); the degenerate exemption set may not grow.
- Unchanged: deterministic outputs and row records, zero-den => zero-num, named numerics change in the commit, all
  four routes, the identity / workspace checks.

Autonomy (Hao): rule questions no longer wait for him: take the most defensible option that protects the zero-sum
identity and the fp32 error, record it here with its evidence, and continue. The goal stands (every main kernel >= 90%
of its calibrated sol4 limit on the four acceptance shapes; first milestone FA3 fp8 forward parity).
- Decided under the autonomy rule (round 17, evidence: the forward helper's C3 / C1 numerics on the default route vs the
  r17_c3 certificate): the 1e-5 per-case tolerance applies to MATCHED cases (the tuned method); stale / full cases
  (correctness only) allow a per-case rise of at most 1e-3, because their stale delta rowsum(dO * O) amplifies any O
  round-off change into dQ / dK (observed: up to +2.1e-4, sign-random, 7 up / 13 down for dK); the per-output MEAN error
  over all cases (all modes) still may not rise, and every other v4 predicate is unchanged.
- Decided under the autonomy rule (round 17), residual-v4 round-off allowance: per row, "accepted + fp32 round-off"
  includes the propagated round-off of the forward's LSE, measured on the ACCEPTED build: two control builds (the
  accepted DBG library and kernel unchanged; attn.py moves the saved LSE by +1 / -1 ulp via torch.nextafter, frozen
  roots r17_p1_dbg_lsep / _lsem) are captured beside the previous DBG build, and a candidate row regresses only above
  accepted + k_r 2^-23 + max(|row(LSE+1ulp) - row(accepted)|, |row(LSE-1ulp) - row(accepted)|) (gate
  `--lse-envelope-roots`). Evidence (forward helper, reproduced with my checker on r17_p1): the unchanged kernel with
  LSE +-1 ulp alone "regresses" 1930 well-conditioned rows under the bare k 2^-23 rule (scale10 301 rows, max +0.29;
  outliers 119; scale3 413), so without it any forward change that moves LSE at round-off fails regardless of accuracy.
  The envelope comes from the accepted build only; the candidate cannot widen it.
- Decided under the autonomy rule: a frozen inherited defect may be REPAIRED (gate `--allow-defect-repair`): the
  candidate's output for that case is finite, with an fp32 error (and every other predicate) not worse; a moved or
  grown defect still fails. Evidence: a 192-row forward tiling changes the all-zero-input NaN pattern; the fixed
  candidate (FLT_MIN floor on the folded scale, bitwise for nonzero scales) gives finite O = 0 for V = 0.
- Decided under the autonomy rule: "the mean over all cases may not rise" allows noise of at most 1e-6 absolute per
  output tensor (one tenth of the per-case tolerance; a systematic drift of the per-case size still fails). Evidence:
  forward C3n / C3f raise the O / dK / dV means by 1e-8 .. 5e-7 with per-case changes of both signs.
- Decided under the autonomy rule (round 17), forward floor: the forward helper proposed floor_fwd = max(sol4 tensor
  floor, measured FA3 fp8 forward time) (evidence: FA3 fp8 runs at 59.0% x 1.163 = 68.6% of the same tensor floor;
  the method's per-group basis + Horner work costs ~+24% over FA3 in kda-fwdport's replica; the forward's issue floor
  is 81% of its tensor floor). NOT adopted as the goal floor: FA3's time measures what a production kernel achieves,
  not work the method needs, and FA3 parity is already the separate first milestone. Adopted instead: (1) every
  accepted round reports BOTH the sol4 % and the FA3-referenced % (forward time vs FA3 fp8 forward time on the same
  shapes); (2) a floor correction is admissible when it counts work the method needs: the per-group P^ basis / Horner
  rho chain / promote instructions are now to be added to the forward's sol4 issue / MUFU floors from the emitted
  step loop (they are in the ledger already; the tensor floor stays the calibrated GEMM peak).
- Decided (round 17): after an accepted, certified NAMED NUMERICS change (forward C3g), byte equality with the frozen
  original baseline is no longer attainable; for bitwise candidates both the "baseline" and the "previous" byte
  references of the full gate are the latest numerics-certified accepted build (r17_fwd dumps, equivalence manifest
  v4, .local/equiv4/r17fwd_*.pt). The original baseline stays the reference of the numerics gate (errors vs fp32) and
  of the progress A/B.
- Decided (round 17), calibration stability: the same-job fp8 GEMM calibration varies ~5% between jobs (1409 vs 1481
  TFLOPS with identical clocks / copy bandwidth), which moves every "% of floor" by the same amount. Each accepted
  round now reports the % against its same-job calibration AND against a fixed reference = the highest fp8 peak
  observed so far (1481.15 TFLOPS, job 21181967; the most demanding floor). Goal attainment is judged on the fixed
  reference.

## Round 17: corrected floor sol4c (Hao, after kda-solcheck)

Evidence (kda-solcheck, origin/kda-sm90-solcheck b5be598, results/solcheck/README.md on that branch): persistent
synthetic kernels that issue exactly the accepted step loops' instruction mixes (same wgmma shapes / counts / SS-RS
sources, same MUFU / FFMA / FMUL / FADD / FMNMX / F2FP / SHFL / LDS counts, filler to the exact instruction count, same
warpgroups and register footprint, no ptxas serialization, random fp8 operands, same-job calibration) reach at most
69.9% (forward), 62.0% (pass 1) and 74.5% (pass 0) of sol4-v2 in the best of five schedules (real dependence order,
no dependence, interleaved, warp-specialized, ping-pong). On this H100 tensor and ALU work couple at about
C_alu + 0.55 C_tensor, the loaded SM clock is 1.66-1.80 GHz and IPC tops out near 3.5. So 90% of sol4-v2 is not
reachable with the current mixes.

Rule: the working target is sol4c = sol4-v2 floor / ideal fraction (forward 0.699, pass 1 0.620, pass 0 0.745; pass 2
and the memory kernels keep their sol4-v2 floor), at >= 90% per kernel on the four acceptance shapes, judged on the
fixed-reference calibration; every report shows both the sol4-v2 % and the sol4c % (benchmarks/sol4.py, IDEAL_FRACTION).
FA3 fp8 forward parity stays the first milestone. Status on r17_pk (job 21181967, 1481.2 TFLOPS reference):
pass 0 94.0% and pass 2 94.1% meet it; forward 78.5% and pass 1 66.3% do not (solcheck's own same-job calibration,
1437 TFLOPS, reads forward ~82%, pass 1 ~68%). Direction (Hao): pass-1 scheduling levers are spent (kda-p1split: freeing
registers for a pipelined S / dP was 3.5-4% slower); pass-1 work removes instructions from the per-element path
(sensitivity ~0.05 sol4 points per instruction per warp-step); forward work targets the gap to FA3 fp8; helpers with
exhausted levers report DONE; pass-0 work stops (it meets the target).

## Round 17: structural + numerics direction (Hao's decision, no further consultation)

1. Structural: pass-1 variants that reuse pass 0's work (pass 0 stores what pass 1 needs, or pass 1 computes the
   matched delta itself in a two-phase schedule so pass 0 disappears), starting from kda-fuse's measured structure
   (origin/kda-sm90-fuse); prototype first, continue if a prototype gains on the whole backward. Gated as usual
   (bitwise full gate, or the residual-v4 numerics gate if the delta's rounding changes).
2. Numerics: the exp clamp and the fp32 dK fold may change as a NAMED numerics change (the certified flag set
   updated in the same commit) when the candidate passes residual-v4 on all four routes; cheaper forms are tried and
   residual-v4 decides.
3. Held stacks are combined across helpers (forward PRETURN + k/v-scale staging with any further forward piece;
   pass-1 SHFL8 with the new pass-1 work) and gated when a combination clears 3% in two jobs.
4. results/REPORT.md is written now as an interim report and updated after every accepted round.
- Decided (round 18), changes that touch two kernels: a single change that alters two kernels (e.g. kda-pass1 C4:
  SHFL8 + the exp clamp kept only on pass 0's masked-tile predicate, affecting pass 0 and pass 1) is gated with the
  kernel that clears 3% as the --target; the other touched kernel must not be slower beyond spread and the e2e fwd+bwd
  must improve. A clamp change that is bitwise on all 72 x 4 certification cases is still recorded as a named numerics
  change (outputs may differ only where the clamp would have bound); the bitwise full gate certifies it.
- Hao (round 19): the two-job rule stands; a candidate that passes one job and misses the other is not rerun as the
  same pair. A near-miss stack is extended with one more held piece (bitwise or residual-v4) so it clears 3% with
  margin, and the larger stack is gated as a fresh pair.

## Round 21: residual-v5 (Hao; replaces residual-v4 for every non-bitwise candidate, own and helpers')

1. No exemptions for speed changes: every matched row, including the rows v4 exempts (residual above 1e-3 in the
   accepted build, degenerate rows), must not exceed its residual in the ORIGINAL kernel (main 3d4cd5b) by more than
   fp32 round-off of that row (k_r 2^-23 relative to the row's own sum of |terms|). The reference is the original, not
   only the previous accepted build, so small regressions cannot accumulate.
2. Measured tensors: the fp32 dS row sums the identity is defined on (before dS is quantized to e4m3), and the delta
   used by the backward against the delta recomputed from the same fp8 P / dP. The exact captured tensors are stated
   below once implemented (benchmarks/numerics.py v5).
3. Coverage: the 72 cases on all four routes; the four acceptance shapes (B2 N8192, B1 N65536 at 16/4 and 32/4, causal
   matched) with >= 3 input seeds; stress inputs (scaled x3 / x10, outliers, near-one-hot rows, long sequences). Per
   case: rows checked, max and median residual, rows worse than the original.
4. Every accepted named numerics change (forward C3g, EXPBX64, C4 masked-line clamp, and any other) is re-audited
   under v5; one that makes any row worse than the original beyond round-off is reverted or fixed before anything
   else is accepted.
5. Bitwise candidates pass automatically; residual-v5 gates every non-bitwise candidate from now on.

## Round 21: docs/kda-sm90/TASK.md adopted (Hao) as the current task for kda-sm90 and every helper

Added as given; where it disagrees with plan.md or older notes, TASK.md wins. Fact fixes / notes on adoption:
- Fixed: AGENTS.md is not in the repo; the cluster rules are ~/workspace/haok1402/agents/AGENTS.md (TASK.md now says so).
- Supersedes earlier error-tolerance decisions: TASK.md's "error against fp32: per case at most 1e-5 relative above
  the ORIGINAL, and the mean over cases does not rise" replaces residual-v4's reference (the accepted build), its
  stale / full 1e-3 per-case allowance and its 1e-6 mean-noise allowance (round-17 decisions). Applied from now on in
  the residual-v5 gate and in the re-audit of the accepted numerics changes.
- New scope requirement: stale mode, full attention, head dim 256 and other lengths must not get slower. The current
  gate does not time them (they were correctness-only since round 12); a no-regression timing check over those
  configurations is being added to the gate.
- Two-kernel changes: "the target kernel (or the summed time of every kernel a change touches)" replaces the round-18
  rule (target the kernel that clears 3%).
- Decided (round 21), residual-v5 row check: "within fp32 round-off of its residual in the ORIGINAL kernel" is the
  round-off of computing that row's sum from its actual terms: per row, no exemptions, |Σt|_cand <=
  |Σt|_orig + k_r 2^-23 S_r with S_r = Σ|P^dP| + |delta_used| ΣP^ of the original's row (t = P^ (dP - delta), P^ already
  carries the 2^8 factor; fp64 sums from the DBG op's rowsx). Evidence (.local/rounds/r21-v5/REPORT.md): the ratio
  form (|Σt| / Σ|t| vs the original + k 2^-23) rejects the ORIGINAL kernel against its own +-1 ulp LSE perturbation on
  4,279 / 4,471 72-case rows (20-30x more than HEAD's 191) because on cancelling rows Σ|t| is 1e-4..1e-2 of S_r; under
  the summation bound HEAD, C3g, EXPBX64 and C4 all pass on the 72 cases x 4 routes, the acceptance shapes (3 seeds)
  and the stress suite (max needed factor 0.954 of the bound; C4 bitwise). The +-1 ulp LSE controls exceed this bound
  (needed factor up to 89): they perturb the input, not the summation, so they do not calibrate it.
  Delta used vs delta recomputed from the same fp8 P / dP: within fp32 round-off (<= 3.5e-7 relative) in every build.
  The error-against-fp32 clause of TASK.md is audited next; it decides C3g / EXPBX64.
- Decided (round 21), the TASK.md error clause and the re-audit (evidence: results/residual_v5_audit.md section f; all
  four routes identical): forward C3g FAILS (absolute reading +1e-5: 12 stale / full case-outputs, worst
  d128_mqa/causal/stale dK +2.1e-4; relative reading x(1 + 1e-5): also 5 matched O cases up to +1.04e-4 relative; the
  per-output mean rises for O / dK / dV); EXPBX64 FAILS (2 stale case-outputs absolute; +1 matched dK +1.2e-5
  relative; dQ / dK means rise); C4 PASSES (bitwise to its predecessor); the fold-free fallbacks (nofold, C3n2) fail
  the same O cases, so the cause is BM192 + FWDLAZY, not the fold. TASK.md: a failing accepted change is reverted or
  fixed before anything else is accepted -> C3g and EXPBX64 are REVERTED to the original arithmetic (release flags:
  FA3FP11_HD128_BM192 = 0, BN256 = 1, FWDLAZY = 0, FWDDFOLD = 0; KDA_PK_EXPB = 0 in pass 0 and pass 1) while every
  bitwise piece stays; the revert is accepted on bitwise equality with the original arithmetic (or the v5 audit) and
  needs no speed gain. Fixes re-enter only as new candidates passing residual-v5 + the error clause.
- Decided (round 21), reading of TASK.md's "per case at most 1e-5 relative above the original": the relative-L2 error
  (vs fp32) of each case / output may exceed the original's by at most 1e-5 in those relative-error units (the reading
  used since residual-v4; the alternative x(1 + 1e-5) reading is reported alongside). C3g and EXPBX64 fail under
  either reading, so the revert does not depend on this choice. The per-output mean over cases must not rise.
