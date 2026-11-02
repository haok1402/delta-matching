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
