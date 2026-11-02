# TASK: kda-sm90 kernel evolution

The current task of every kda agent working on delta-matching. Where plan.md (frozen at the start of the RLCR loop) or
an older note disagrees, this file wins. Decisions and their evidence stay in rule-change-r10.md; results in rounds.md
and REPORT.md.

## Goal

Speed up the matched-delta method (Delta-Matching, arXiv 2609.37852) on H100 without weakening its numerics.

- Kernels: forward, pass 0 (matched delta), pass 1 (dK, dV, dS'), pass 2 (dQ), plus the producers and glue they need.
- Target: each of forward, pass 0, pass 1 and pass 2 at >= 90% of its corrected floor sol4c (sol4-v2 divided by the
  ideal-mix fraction measured by kda-solcheck; pass 2 keeps its bytes floor), on the fixed-reference calibration.
- First milestone: the forward at least as fast as FA3's fp8 forward on the acceptance shapes.
- Report both % of sol4-v2 and % of sol4c every accepted round.

## Scope

- Head dim 128, causal, matched mode. Stale mode, full attention, head dim 256 and other lengths must stay correct
  and must not get slower, but are not tuned.
- Acceptance shapes (the paper's training shapes plus Qwen3-30B-A3B attention), equal weight, geometric mean:
  - B2 N8192 and B1 N65536 at Hq 16 / Hk 4 (the paper's 1.67B GDN/GQA hybrid);
  - B2 N8192 and B1 N65536 at Hq 32 / Hk 4 (Qwen3-30B-A3B).
- Hardware: cwpark H100 through Slurm only (rules in ~/workspace/haok1402/agents/AGENTS.md). Ncu counters are blocked for non-admins; use timing
  ablations, what-if builds, SASS counts and the solcheck microkernels.

## Acceptance

A change is accepted when all of these hold:

- pytest passes in the identity-verified candidate process.
- Numerics:
  - bitwise equal to the accepted build on 72 cases x 4 routes (default, kv, head, streams2); or
  - a named numerics change that passes residual-v5 (below).
- Speed: the target kernel (or the summed time of every kernel a change touches) >= 3% faster on the acceptance shapes,
  in two fresh jobs; no acceptance shape slower beyond its run-to-run spread; bench.py within 1%.
- Resources: no ptxas spill or serialization diagnostics (stack growth is reported, not fatal).
- Small gains: bitwise or residual-v5 gains >= 1%, confirmed by two jobs, are held and stacked until a stack clears 3%.
  A near-miss stack is extended and gated as a fresh pair, never rerun as is.

## Numerics: residual-v5 (the zero-sum identity never gets worse)

- Measured on the fp32 dS row sums before dS is quantized, and on the delta used against the delta recomputed from the
  same fp8 P / dP.
- Every row, with no exemptions, stays within fp32 round-off of its residual in the ORIGINAL kernel (main 3d4cd5b).
  "fp32 round-off of the row" is the round-off of computing its sum from its own terms: |sum t|_cand <=
  |sum t|_orig + k 2^-23 (sum |P^ dP| + |delta| sum P^) (benchmarks/v5; decided round 21, rule-change-r10.md).
- Error against fp32: per case at most 1e-5 relative above the original (relative-L2 error units; both readings
  reported), and the mean over cases does not rise. A repaired inherited defect (the original's NaN becoming finite
  and correct, error not larger anywhere) is allowed.
- Deterministic outputs; the debug build equals the production build.
- Coverage: 72 cases x 4 routes, the four acceptance shapes with at least three seeds, and stress inputs (scaled x3 and
  x10, outliers, near-one-hot rows, long sequences).
- Every numerics change already accepted is re-audited under residual-v5 and reverted or fixed if it fails.

## Process

- kda-sm90 is the integrator: it alone gates and merges into kda-sm90 and runs its own rounds in parallel. Helpers
  work on their own branches (kda-sm90-<name>) in their own worktrees and post candidates in CANDIDATES.md.
- Search widely: many short parallel screens first, depth on what screens well; log every screened variant with its
  number. A helper whose levers are exhausted says DONE with its evidence.
- Do not wait on Hao: decide within these rules, record the decision with evidence in rule-change-r10.md, keep going.
- Git: commit as Hao Kang through the repo config, push only to kda-sm90 and kda-sm90-* on haok1402/delta-matching,
  never to main or upstream, never force-push.
- Results JSON stays out of git (~/workspace/haok1402/kda-results/), except results/progress/A_median.json and
  B_median.json. Numbers that matter go in the markdown files.
- After every accepted round: update REPORT.md, the sol4 / sol4c table, and the original-vs-now A/B in
  results/progress/.
