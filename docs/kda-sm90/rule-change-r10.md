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
