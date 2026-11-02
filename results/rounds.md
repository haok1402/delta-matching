# Rounds log (kda-sm90)

Every kernel hypothesis, accepted or not. Gate: pytest; byte equivalence against the frozen baseline
(`benchmarks/equivalence.py`, four routes); >= 3% on the target kernel's affected main-grid subset in a paired,
order-alternated A/B (`benchmarks/ab.py`, 3 reps), confirmed by a second fresh job; no shape over +1% fwd+bwd beyond its
own repetition spread (main and held-out grids); bench.py D128 / D256 not worse than 1%; no new spills / stack /
serialization.

Stop rule (Hao, after round 1): per kernel. After two consecutive failed profile-based rounds on one kernel, move to the
next kernel in the waste ranking; stop when each of the top four kernels (pass 1, forward, pass 2, torch glue) has two
consecutive failures or is within 10% of its modeled limit. Non-bitwise changes are allowed when the numerics contract
holds (AC-2.3). Hypotheses are based on a profile of the kernel.

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 0 | 1 (H1a, round 1) |
| forward | 0 | 0 |
| pass 2 | 0 | 0 |
| torch glue | 0 | 0 |

Only a rejected profile-based candidate increments its kernel's eligible counter; an accepted candidate resets it;
profiling / tooling rounds change neither.

## Round 1: H1a, pass 1 hd128, dV gemm issued before the cross-warpgroup psi barrier: REJECTED

- Change (`results/r1/h1a.patch`, `csrc/mainloop_bwd_sm90_fp8r.hpp`): after the dS line (the last reader of the fp32 P
  fragment), the P~ conversion, the dV window rescale / f_prev update, the dOt wait and the dV RS wgmma are issued before
  the psi amax / named barrier; the dK wgmma is issued after the dS' conversion as before; FIFO retirement unchanged
  (wait<1> retires dV then releases dOt, wait<0> retires dK). wgmma ledger per tile: S, dP (wait<1>, wait<0>), dV,
  dK (wait<1>, wait<0>), same groups and order as the baseline, dV issued earlier.
- Resources: REG 168, STACK 0 (causal) / 8 (non-causal), LOCAL 0 for the two pass-1 instantiations: identical to the
  baseline library (which also has STACK 8 on the non-causal instantiation); no ptxas spill / serialization
  diagnostics.
- Correctness: pytest 49 passed / 3 skipped; equivalence vs the frozen baseline, default route, 72 cases: BITWISE
  EQUAL (library sha256 36f86c61... vs 07fee0f2...).
- Performance (job 21157817, `results/r1/ab_main_full`, `results/r1/ab_holdout_full`): pass 1 on the D128 subset
  245.518 -> 242.412 ms (-1.27%, spreads 0.43% / 0.60%); pass 1 over the whole main grid -0.43%; e2e fwd+bwd D128
  -0.61%; held-out pass 1 -0.51%, e2e -0.20%, no held-out shape over the gate; two main shapes at +1.14% / +1.04%,
  both within their own repetition spread. bench.py N8192 matched fwd+bwd: D128 4.524 / 4.481 -> 4.460 / 4.448 ms,
  D256 8.750 / 8.724 -> 8.778 / 8.684 ms.
- Verdict: below the 3% gate on the target subset; reverted. Reading: hiding the psi barrier wait behind the dV gemm
  recovers about 1% of pass 1, so the per-tile cross-warpgroup synchronization is not pass 1's main cost; the
  per-element issue work (exp2, dS, conversions, the dS' transpose) is the more likely limiter.

## Round 2: pass-1 profile (no kernel change)

Nsight Compute is unavailable (RmProfilingAdminOnly=1 on the nodes), so pass 1 was profiled with a clock64 probe build
(`DM_PROF=1`, bitwise to the baseline, ~+20% overhead) plus what-if builds: `results/profile_pass1.md`, raw data in
`results/r2/`. Findings: matrix work at modeled peak is 23-26% of the instrumented step time (issue pressure and serialization are
hypotheses consistent with this, not measured), the MUFU exp2 floor is half the tensor ideal at hd128, the d256 single dO stage exposes
load latency (a second dO stage: -2.4 .. -5.0% d256 pass 1 in two fixed-order exploratory repetitions, bitwise), and
deeper hd128 stages are slower. Issue pressure and warpgroup lockstep are hypotheses consistent with the data (no
hardware counters, no per-step timeline yet); the local-psi timing probe is not an isolated bound. Ranked profile-based hypotheses
are in the profile report; the next round takes P1-d256-DO2.

## Round 3: P1-d256-DO2, pass 1 d256 dO double-buffered: ACCEPTED

- Hypothesis (profile-based, results/profile_pass1.md #1): the d256 pass 1 exposes dO load latency with one dO stage.
- Change: `csrc/flash_bwd_fp8v10_launch.h` `Stages_dO_d256` 1 -> 2 (D128 untouched). With Q and dO both at 2 stages
  the collective takes its `Q_dO_same_stages` path: the dO pipeline uses the Q pipeline's state (same stage index and
  phase; the producer's separate dO barriers carry the same transaction byte count, since the 64 x 256 e4m3 dO tile and
  its dPsum match the Q tile and its LSE), no separate dO state increments; both wrap every 2 m-blocks. Dynamic smem
  212,992 -> 229,376 B (+16 KB dO stage), + 1 KB static, under the 232,448 B limit.
- Resources: d256 pass-1 instantiations REG 168 / LOCAL 0, STACK unchanged or lower (8 -> 0 on one), no ptxas spill or
  serialization diagnostics (results/r3/build_resources_pytest.txt, resource_*.txt).
- Correctness: pytest 49 passed / 3 skipped; captured-intermediate equivalence (manifest v3, schema-validated) vs the
  frozen baseline: BITWISE EQUAL, FULL SUITE on default (72 cases), kv, head, streams2 (60 each); baseline vs checkout
  bitwise on all four routes (determinism across processes); a flipped byte in a captured per-group delta is
  reported (results/r3/equivalence_captured.txt, job 21159224).
- Performance, job 21158945 (3 alternating reps, conflicting DM_SO inherited and ignored): D256 pass 1 262.969 ->
  253.443 ms (-3.62%) main, 26.116 -> 25.195 ms (-3.53%) held-out; D256 e2e fwd+bwd -1.72% / -2.23%; shapes over +1%:
  a D128 main shape +2.14% driven by one 3.81 ms B repetition (within its spread; D128 kernels are unchanged) and a
  D128 held-out shape +1.08% (within spread).
- Confirmation, fresh job 21159225: D256 pass 1 258.644 -> 249.231 ms (-3.64%) main, 25.805 -> 24.834 ms (-3.76%)
  held-out; all-grid pass 1 -2.12% / -2.14%; e2e fwd+bwd D256 -1.80% / -1.61%; no shape over the +1% gate on either
  grid. bench.py N8192 matched fwd+bwd (3 alternating reps): D256 8.623 / 8.635 / 8.616 -> 8.468 / 8.492 / 8.463 ms,
  D128 4.474 / 4.466 / 4.464 -> 4.461 / 4.472 / 4.483 ms (results/r3/bench_*.txt).
- Counters: pass 1 eligible 0 -> 0 (accepted: reset).

DO2 acceptance coverage completed in round 4: the forced routes now run all 72 cases (the earlier route filter skipped
the 12 MHA minimum-length / packed cases); baseline vs DO2 BITWISE EQUAL, FULL SUITE, 72 cases on kv, head and streams2
(job 21159927, results/r4/equivalence_b6_prof.txt), in addition to the default route's 72 (round 3).

## Round 4: pass-1 sparse timeline (profiling; no kernel change, counters unchanged)

A timeline-only build (`DM_TRACE=1`) records absolute clock64 event times for 8 sampled CTAs x 12 m-block steps x 3
actors (two consumer warpgroups, producer) in both pass-1 collectives; bitwise to the DO2 build (72 cases); +14-16%
pass-1 overhead (dense probes +17-26%). Results in `results/timeline_pass1.md` (raw trace and summaries in results/r4/):
in the sampled, instrumented records the consumer load-wait regions are short (60-100 cycles) and the producer
acquire regions long (880-2900 cycles per step); the bracket from the dV retire/release marker to the dK retire/release/state marker is ~220-255 cycles per
step (from the dK issue marker: 269 / 280 cycles at D128 causal / full, 247 / 250 at D256); vector work is ~2000-2600 of ~3850
cycles per warpgroup step; the warpgroups start each step within ~60 cycles of each other and only 4-40% of one's vector
work runs under the other's wgmma. Next pass-1 hypothesis: H1b (task4 review first).

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 0 | 1 (H1a, round 1) |
| forward | 0 | 0 |
| pass 2 | 0 | 0 |
| torch glue | 0 | 0 |

## Round 5: H1b, pass 1 hd128, dS' transpose / store deferred under the dK gemm: REJECTED (profile-based)

- Hypothesis (timeline-based, results/timeline_pass1.md): the dK tail (269 / 280 cycles from the dK issue marker to
  the dK retire/release/state marker at D128) can absorb the dS' transpose / stmatrix / TMA store.
- Review first (task4, results/r5/task4_codex_review.md): SAFE at the source / PTX level (no access to rdS / rP /
  accumulators while outstanding; dV / dK retirement before dOt / Qt release and the fold; the sdS reuse sequence
  intact); register budget near 240, so the 8 packed words are overwritten in place.
- Change (results/r5/h1b.patch): the 8 natural dS' words are packed from rdS before the dV / dK issues; after the dK
  issue the shuffles, stmatrix, WG barriers, proxy fence and TMA store run from those words.
- Resources: identical to DO2 (REG 168, STACK 8 / 0, LOCAL 0; QGMMA 48, DEPBAR 16, ARRIVE 16; +16 SASS lines), no ptxas
  spill / serialization diagnostics.
- Correctness (job 21161224): pytest 49 / 3; captured-intermediate equivalence BITWISE EQUAL, FULL SUITE, 72 cases on
  default / kv / head / streams2 vs BOTH the frozen baseline and the accepted DO2 revision.
- Performance (job 21161224, A = DO2, B = H1b, 3 alternating reps): D128 pass 1 248.145 -> 244.075 ms (-1.64%) main,
  20.626 -> 20.136 ms (-2.38%) held-out; one D256 main shape +3.05% within its spread (D256 kernels unchanged).
  bench.py N8192 matched fwd+bwd D128 4.505 / 4.523 / 4.618 -> 4.487 / 4.479 / 4.544 ms.
- Verdict: below the 3% gate; not merged (developed in a worktree). How many cycles of the instrumented dK bracket the
  schedule recovered is not measured (kernel-level timing vs a perturbed diagnostic bracket). Placement: in the emitted
  SASS of the causal masked path all 16 transpose shuffles run before the first dK QGMMA (under the outstanding dV),
  other paths interleave them with the dK K-loop; the dS' stmatrix / TMA store do precede the group waits. The
  measurement therefore tested the compiler-selected overlap, not every shuffle after the dK issue.
- Counters: pass 1 eligible 0 -> 1.

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 1 (H1b) | 1 (H1a, round 1) |
| forward | 0 | 0 |
| pass 2 | 0 | 0 |
| torch glue | 0 | 0 |

Next (Hao, round 5): pass-1 warpgroup ping-pong with per-warpgroup psi, then partial FMA-polynomial exp2 (both
non-bitwise: the Q2 numerics gate and Q3 workspace capture are their acceptance prerequisites).

## Round 6: PINGPONG (pass 1 hd128): per-warpgroup psi + warpgroup ping-pong of the S / dP issue: REJECTED at stage 1

- Hypothesis (Hao's direction after the timeline): staggering the two warpgroups so one's vector work runs under the
  other's wgmma shortens the step; it needs per-warpgroup psi (no cross-warpgroup psi rendezvous).
- Staged gate (round-6 contract): stage 1 builds the real pass-1 side of the design and times the gated metric; stage 2
  (pass 2 two-scale consumption, Q2 numerics gate, Q3 workspace capture, full gate) only if stage 1 reaches 3%. Pass 2's
  changes can only add cost to the candidate, so stage 1 bounds the pass-1 gain. Stage 1's numerics are intentionally
  incomplete (pass 2 still reads one psi per tile): timing only.
- Change (results/r6/pingpong_stage1.patch, worktree build, `KDA_PINGPONG`): psi per 64-key warpgroup half from the
  warpgroup's own four warp slots behind a WG-local barrier (ids 5 / 6, already the warpgroup's transpose barriers),
  published as psi[BH, nbm, nbn, 2] (allocation doubled at hd128); ping-pong turn barriers ids 2 / 3 (dQEmptyWG1 / 2,
  unused by P1-R): each warpgroup syncs its turn (256 threads) before its S / dP issue and arrives on the other's right
  after the dP issue; warpgroup 1 pre-arrives warpgroup 0's turn once per CTA (the arrival left at each tile's end is the
  next tile's prologue; both warpgroups run the same m-blocks, so per-tile syncs and arrives balance).
- Emitted code (results/r6/pingpong_barrier_placement.sass, gate.json): the 256-thread turn BAR.SYNC sits directly
  before each S QGMMA batch and BAR.ARV directly after the dP QGMMAs; three 128-thread barriers per step (psi + the two
  transpose barriers); BAR.SYNC 18 -> 22, BAR.ARV 2 -> 8; QGMMA 48 unchanged; REG 168, STACK 8 -> 0, LOCAL 0, no CALL;
  no ptxas spill / serialization diagnostics.
- Measurement (job 21161717, benchmarks/gate.py --mode timing, A = DO2, B = stage 1, 3 alternating reps): D128 pass 1
  243.398 -> 247.692 ms (+1.76%, slower) main, 20.416 -> 20.768 ms (+1.72%) held-out; e2e D128 fwd+bwd +0.75% on both
  grids; bench.py D128 matched fwd+bwd 4.471 / 4.490 / 4.462 -> 4.513 / 4.513 / 4.510 ms.
- Verdict: REJECTED at stage 1 (slower on the target kernel); not merged; stage 2 not built. Reading: forcing the
  warpgroups to alternate their S / dP issue serializes them more than it overlaps their vector work with the other's
  wgmma. (Earlier exploratory evidence, round 2: per-warpgroup psi WITHOUT the turn barriers was -4.2 .. -4.8% on pass 1;
  that variant is a different hypothesis: decoupling only.)
- Counters: pass 1 eligible 1 -> 2.

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 2 (H1b, PINGPONG) | 1 (H1a, round 1) |
| forward | 0 | 0 |
| pass 2 | 0 | 0 |
| torch glue | 0 | 0 |
