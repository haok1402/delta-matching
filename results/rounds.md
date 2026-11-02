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

## Campaign transition after round 6: pass 1 closed, forward next

Pass 1 has two consecutive failed profile-based rounds (H1b, PINGPONG) and is closed under the stop rule. The partial
FMA-polynomial exp2 (MUFU-poly) and decoupling-only per-warpgroup psi stay as unexecuted pass-1 backlog for the report.
Current ranking from the accepted DO2 arm of round 6 (results/r6 ab_main A_median.json; main grid sums, modeled SOL from
`kernels.accounting`):

| kernel | measured ms | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|
| pass 1 | 493.384 | 128.810 | 364.574 | 26.1% (closed) |
| forward | 185.745 | 64.167 | 121.578 | 34.5% |
| pass 2 | 105.895 | 63.061 | 42.834 | 59.6% |
| torch glue | 36.473 | 13.822 | 22.650 | 37.9% |
| pass 0 | 35.675 | 16.042 | 19.633 | 45.0% |

The order is unchanged since round 3. Forward is the largest eligible gap: the next campaign profiles it (round 7).

## Round 7: forward profile, then PVIL (forward, both head dims): interleaved half-width PV temp chains: REJECTED

- Profile (`results/timeline_fwd.md`, `results/forward_design_diff.md`, `results/r7/`): the forward sparse trace is a
  failed diagnostic (+5 .. +48% overhead, and its forward O / LSE differ from DO2 on every case family: the probes change
  code generation), so the profile rests on static SASS against FA3's fp8 forward and on timing-only ablations (job
  21163152). Largest measured cost: the rho rescale between the serialized PV sub-gemm groups, -19.8 .. -20.2% of the
  D256 forward and -7.7 .. -8.1% of D128 when removed; S descale -2.9 .. -9.0%; exposed k-scale loads -0.8 .. -3.2%
  (the only knob expected bitwise, not equivalence-tested, below 3%).
- Hypothesis PVIL (from that profile): run the PV temp as two half-width temps driven as interleaved rho chains (each
  temp's rescale under the other's sub-gemm). D128 NSPLIT 1 -> 2 (one pair of 32-register temps), D256 NSPLIT 2 -> 4
  (two pairs); PV-phase registers unchanged (D128 160, D256 208 fp32 / thread). Certified flag set changed in the
  candidate: FA3FP8F_TP_NSPLIT 2 -> 4, FA3FP11_HD128_NSPLIT 1 -> 2, FA3FP11_FWD_PVIL 0 -> 1 (`results/r7/pvil/pvil.patch`).
- Gate (job 21163612, `benchmarks/gate.py --mode full`, all steps rc 0, `results/r7/pvil/`): tooling tests pass;
  candidate pytest 49 passed / 3 skipped inside the identity-verified process; **bitwise equal on all four routes (72
  cases each) against the frozen baseline and against DO2**; resources: 4 forward functions on both sides, REG 168,
  LOCAL 0, STACK 24 / 24 / 24 / 32 -> 24 / 24 / 24 / 40, CALL 0, no spill / serialization diagnostics; QGMMA 140 -> 220,
  WARPGROUP.DEPBAR 70 -> 130, STL 35 -> 53, LDL 59 -> 75 (all four instantiations); the emitted PV phase has two
  `QGMMA.64x64x32` groups in flight before each `WARPGROUP.DEPBAR` (`pv_order_*_d256_full.txt`).
- Measurement (A = DO2, B = PVIL, 3 alternating reps): forward main grid 186.763 -> 183.875 ms (**-1.55%**; D256
  -3.03%, D128 -0.08%), held-out 17.175 -> 17.001 ms (-1.01%; D256 -2.57%); e2e fwd+bwd -0.11% main / -0.17%
  held-out; four D128 main shapes +1.04 .. +1.54% fwd+bwd, all within their repetition spreads; held-out none over 1%;
  bench.py matched forward D128 1.097 / 1.093 / 1.095 -> 1.080 / 1.082 / 1.080 ms, D256 2.018 / 2.020 / 2.020 ->
  1.999 / 1.999 / 2.002 ms.
- Verdict: REJECTED (the change affects the forward at both head dims; -1.55% on its affected sum, below 3%); not
  merged. Reading: interleaving two chains does not shorten either chain; each temp still runs four dependent
  sub-gemm -> wait -> rescale stages (a reading of the emitted order, not a measured attribution). The knob-2 / 3
  removal timings size the remaining opportunity; independent group temps are one way to approach it (accumulation
  order changes: non-bitwise, needs the numerics gate Q2 / Q3), not a proven or exclusive one.
- Resources per symbol (review correction): the round-7 gate printed sorted stack lists, which hid the correspondence.
  Per symbol, D256 causal STACK 24 -> 40 B (new STL / LDL inside the steady-state loop), D256 full 32 -> 24 B; aggregate
  STL / LDL 35 / 59 -> 53 / 75. Absence of ptxas warnings is not an AC-3.3 pass; the gate now checks per-symbol STACK /
  LOCAL growth and fails full mode on it.
- Counters: forward eligible 0 -> 1.

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 2 (H1b, PINGPONG): closed | 1 (H1a, round 1) |
| forward | 1 (PVIL) | 0 |
| pass 2 | 0 | 0 |
| torch glue | 0 | 0 |

## Round 8: PVIL-D256 (Hao's ruling) and GRPFOLD (the rho lever): both REJECTED; forward closes

User direction at the start of the round (Hao): (1) a D256-only PVIL is a fair candidate gated on the D256 forward
subset, like DO2; (2) make the rho rescale between the serialized PV sub-gemms the next forward hypothesis.

Gate change first (B7, 51587fc): every selected target symbol must have one ELF `.text` section, one resource row and
one SASS record with unique contiguous 16-byte offsets from 0 to the exact section end; per-symbol STACK / LOCAL growth
is recorded and fails full mode (the acceptance predicate "no new spills / stack").

### PVIL-D256 (`results/r8/pvil256/`, job 21164633)

- Change: round-7 PVIL with `kPVIL` restricted to kHeadDimV >= 256 (FA3FP8F_TP_NSPLIT 2 -> 4, FA3FP11_FWD_PVIL 1;
  hd128 flags and code path unchanged). Two pre-declared variants (A as round 7; B with rho_1..3 computed once per
  tile), the first without per-symbol stack / local growth to be gated.
- Result: both variants grow the D256 causal (dynamic-scheduler) forward symbol's stack: 24 -> 40 B (A), 24 -> 48 B
  (B), with STL / LDL inside the steady-state loop (round-7 PVIL binary: STL at 0x4b.., 0xa0c0 ..). The gate ran on A
  and stopped at the resource predicate (`FAILED: per-symbol stack / local growth ... STACK 24 -> 40`) before pytest /
  A/B. Timing evidence for this code at D256 is round 7's: -3.03% main, -2.57% held-out D256 forward.
- Verdict: REJECTED (new loop spills in a changed symbol; AC-3.3). Counted with round-7 PVIL as one hypothesis.

### GRPFOLD (`results/r8/grpfold/`, job 21164677)

- Change: independent group PV accumulators. Products k = 0 .. 4*NSPLIT-1 (slab k/4, group k%4), each into a
  zero-initialised half-width temp T[k%2] (32 fp32 regs); ledger per step: issue(k) -> `wait<1>` (retires k-1) ->
  fold O_slab(k-1) += T * w_g with w_g = v8sc_g * 2^(ms_g - m*l2) (softmax.basis<g>), ascending group order per slab;
  the last product stays in flight to the shipped `wait<0>` + promote_last; no Horner rho. Also applied to the tail
  tile. D128 NSPLIT 1 -> 2, D256 2 -> 4. Register plan, PV phase: D128 tOrO 64 + 2 x 32 + P 32 + w 8 = 168; D256 128 +
  64 + 16 + 8 = 216 (shipped 160 / 208). Accumulation-order class (not bitwise). Declared footprint: the forward at both
  head dims (all-grid `fwd`).
- Two pre-declared weight variants (w once per tile / per fold): neither grows any symbol's STACK or LOCAL; the
  per-tile variant was gated (`--mode timing`, stage 1). Resources per symbol unchanged (REG 168, STACK 24 / 24 / 32 /
  24), STL / LDL 35 / 59 -> 33 / 57, QGMMA 140 -> 220.
- Stage 1 (A = DO2, B = GRPFOLD, 3 alternating reps): forward main 185.746 -> 185.969 ms (**+0.12%**; D256 -2.18%,
  D128 slower), held-out 17.168 -> 17.133 ms (-0.21%; D256 -2.41%); e2e fwd+bwd +0.29% main; four D128 causal main
  shapes +1.33 .. +1.90% fwd+bwd beyond their spreads; bench.py matched forward D128 1.088 / 1.091 / 1.094 -> 1.145 /
  1.147 / 1.144 ms, D256 2.004 / 2.018 / 2.024 -> 1.964 / 1.951 / 1.960 ms.
- Verdict: REJECTED at stage 1 (no gain on its declared footprint); numerical certification (Q2 / Q3) not built for a
  candidate that cannot ship. Reading: like PVIL, it keeps the same number of per-element vector operations (four
  FMA folds instead of three rescales + one promote) and only reorders them; neither reordering recovers the knob-2
  removal timing.

### Counters and ranking

Forward eligible 1 -> 2 (PVIL incl. its D256 variant; GRPFOLD): forward has reached the stop condition. Hao's other
rho-lever ideas (folding rho into the P quantization or the V-side scale; deferring the rescale) and the S-descale fold
are recorded as unexecuted forward backlog. Next: pass 2 (largest remaining eligible waste), profile first.

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 2 (H1b, PINGPONG): closed (reopens only with a new profile-based hypothesis, Hao) | 1 (H1a, round 1) |
| forward | 2 (PVIL / PVIL-D256, GRPFOLD): closed | 0 |
| pass 2 | 0 | 0 |
| torch glue | 0 | 0 |

Current ranking (accepted DO2 arm of the round-8 GRPFOLD A/B, `results/r8/grpfold/ab_main/A_median.json`, main grid):

| kernel | measured ms | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|
| pass1 | 493.298 | 128.810 | 364.487 | 26.1% |
| fwd | 185.746 | 64.167 | 121.579 | 34.5% |
| pass2 | 105.897 | 63.061 | 42.836 | 59.5% |
| glue | 36.474 | 13.822 | 22.652 | 37.9% |
| pass0 | 35.804 | 16.042 | 19.762 | 44.8% |

## Round 9: pass 2 profile, then P2ST3 (hd128 pass-2 TMA depth 2 -> 3): ACCEPTED (6a870e9)

- Profile (`results/timeline_pass2.md`, `results/r9/wi/`, job 21165303): emitted order, resources / occupancy and
  timing-only ablations of both pass-2 kernels (no in-kernel probes as timing evidence). Each CTA (one 128-thread
  warpgroup, two CTAs per SM) runs one serial per-tile chain; the psi load is issued after the TMA barrier wait.
  Ablations (8 configs, pass-2 sum): psi constant -4.31%, no in-window rescale -3.63%, hd128 depth 3 -4.49% (-7.8 ..
  -9.0% on the hd128 configs), PSIPF (psi one tile early, bitwise on the default route) -3.09%.
- Hypothesis P2ST3 (declared in the profile with footprint, budget and ledger): `FA3FP11_HD128_P2ST` 2 -> 3
  (certified flag-set change), footprint the D128 pass 2. Smem 83968 -> 108544 B per CTA, two CTAs per SM at both
  depths; registers unchanged; the ring ledger keeps the unconditional release barrier before every re-arm.
- Gate (job 21165579, all steps rc 0, `results/r9/p2st3_1/`): pytest 49 / 3 skipped in the identity-verified process;
  **bitwise on all four routes vs baseline and DO2**; per-symbol resources unchanged, complete SASS. D128 pass 2
  55.653 -> 51.482 ms (**-7.50%**) main, 5.032 -> 4.609 ms (-8.41%) held-out; e2e fwd+bwd -0.23%; shapes over +1%
  within spread; bench.py unchanged within noise.
- Confirmation (job 21165963, same candidate sha d2eee98774eb vs DO2 7d491ed1a556, `results/r9/p2st3_2/`): D128 pass 2
  55.797 -> 52.112 ms (**-6.60%**) main, 5.040 -> 4.642 ms (-7.89%) held-out; one D256 shape +1.34% within spread;
  bench.py D128 matched fwd+bwd 4.502 / 4.505 / 4.498 -> 4.462 / 4.469 / 4.465 ms.
- Verdict: ACCEPTED. New accepted build: library d2eee98774eb (`.local/frozen/r9_p2st3`, stamp code_sha 4b325343a46e
  = the committed source); its route dumps are byte-identical to DO2's. Pass-2 counter stays 0 (reset by acceptance).

| kernel | profile-based consecutive failures (eligible counter) | historical (pre-profile) rejections |
|---|---:|---|
| pass 1 | 2: closed (reopens only with a new profile-based hypothesis, Hao) | 1 (H1a) |
| forward | 2: closed | 0 |
| pass 2 | 0 (P2ST3 accepted) | 0 |
| torch glue | 0 | 0 |

Current ranking (accepted P2ST3 arm of the confirmation, `results/r9/p2st3_2/ab_main/B_median.json`, main grid):

| kernel | measured ms | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|
| pass1 | 501.860 | 130.755 | 371.105 | 26.1% |
| fwd | 188.803 | 65.162 | 123.641 | 34.5% |
| pass2 | 102.914 | 63.061 | 39.853 | 61.3% |
| glue | 36.795 | 13.822 | 22.972 | 37.6% |
| pass0 | 36.506 | 16.291 | 20.215 | 44.6% |
Next pass-2 hypothesis: PSIPF (psi loaded one tile early, both headers), measured against P2ST3.
