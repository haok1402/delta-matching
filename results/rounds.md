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
  (certified flag-set change), footprint the D128 pass 2. Dynamic smem 84992 -> 109568 B per CTA (+1024 B static; corrected from the
  83968 / 108544 B array payloads first quoted), two CTAs per SM at both depths; registers unchanged; the ring ledger keeps the unconditional release barrier before every re-arm.
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

## Rule change (Hao, during round 10): D128 is the target, one grid

From round 10 on (docs/kda-sm90/rule-change-r10.md; enforced by `ab.py decision()` and
`gate.py --target`): no held-out split (one grid of all 78 configurations, `kernels.py --grid all`); the target is the
D128 kernels; a candidate is accepted when its target kernel is >= 3% faster summed over all D128 shapes in the full
gate job AND a fresh confirmation job, no D128 shape is slower beyond its run-to-run spread (median slowdown larger
than the larger repetition range), and D256 stays correct and not slower (no D256 shape slower beyond spread, the
target's D256 sum not slower beyond the spread of its per-repetition sums). D256 is not tuned and needs no
counterpart. Pass 1 and forward reopen at D128 with fresh counters; pass 2 and glue keep theirs. Rounds 3 (DO2) and 9
(P2ST3) stay committed.

| kernel (D128 target) | eligible counter from round 10 |
|---|---:|
| pass 1 | 0 (reopened) |
| forward | 0 (reopened) |
| pass 2 | 0 |
| glue | 0 |

## Round 10: PSIPF (pass 2, both head dims): ACCEPTED (00d951b); P2ST3 re-read under the new rule

- Hypothesis PSIPF (declared from the round-9 profile, contract round 10): psi of tile nb+1 loaded during iteration
  nb, before tile nb+1's TMA barrier wait, both headers; values / predicates / state / windows / accumulation order /
  epilogue / the unconditional release barrier unchanged. Emitted: `@!P0 LDG.E` psi[nb+1] before the `TRYWAIT`
  (`results/r10/psipf_1/code/order_cand.txt`). Resources: REG +0..2, STACK / LOCAL 0.
- Gate job 21166328 (full, vs P2ST3, split grids as run): bitwise on all four routes vs the baseline and P2ST3;
  re-analysed under the new rule (`results/r10/rule/rule_psipf_job1.json`, main + former held-out runs = the merged
  grid): D128 pass 2 56.099 -> 54.296 ms (**-3.21%**), D256 pass 2 -3.52%, no shape slower beyond spread: ACCEPT.
- Confirmation job 21166727 (`gate.py --mode confirm --target pass2`, merged grid, same candidate e1119aaaa89c vs
  P2ST3 d2eee98774eb): D128 pass 2 **-3.14%**, all predicates pass; bench.py D128 matched fwd+bwd 4.495 / 4.454 /
  4.466 -> 4.451 / 4.451 / 4.444 ms, D256 within noise.
- Verdict: ACCEPTED. Accepted build: library e1119aaaa89c (`.local/frozen/r10_psipf`, stamp code_sha 2c31b6253617).
- P2ST3 under the new rule (re-analysis of its two committed jobs, `results/r10/rule/`): job 2 ACCEPT (D128 pass 2
  -6.71%); job 1 D128 pass 2 -7.57% but two D256 shapes register as slower beyond spread although P2ST3 leaves every
  D256 kernel byte-identical (its SASS diff is the hd128 pass 2 only): measurement noise that the per-shape predicate
  can still flag. P2ST3 stays committed (Hao).

Current D128 ranking (accepted PSIPF arm of the confirmation, `results/r10/psipf_2/ab_all/B_median.json`, all D128
shapes of the merged grid):

D128 shapes: 39 configurations

| kernel (D128) | measured ms | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|
| pass1 | 265.646 | 70.540 | 195.106 | 26.6% |
| fwd | 102.105 | 35.044 | 67.062 | 34.3% |
| glue | 22.434 | 7.892 | 14.542 | 35.2% |
| pass0 | 21.693 | 8.761 | 12.931 | 40.4% |
| pass2 | 54.316 | 45.112 | 9.204 | 83.1% |
| prod_fwd | 8.242 | 5.862 | 2.379 | 71.1% |
| prod_bwd | 4.137 | 2.973 | 1.165 | 71.8% |
| twin | 3.119 | 2.731 | 0.389 | 87.5% |
| kernel (D128 target) | eligible counter |
|---|---:|
| pass 1 | 0 (reopened) |
| forward | 0 (reopened) |
| pass 2 | 0 (PSIPF accepted; now 83% of its modeled D128 limit) |
| glue | 0 |
| pass 0 | 0 |

Next: pass 1 at D128 (largest D128 waste), a new hypothesis from the existing D128 profile.

## Direction after the independent audit (Hao, during round 11)

Recorded in full in `docs/kda-sm90/rule-change-r10.md` (plan.md is frozen for the loop): stacking of >= 1% rejected
changes with identical outputs into one >= 3% candidate (first: H1a + H1b (+ AMAX-REDUX) as one D128 pass-1
candidate); build the Q2 / Q3 numerics gate now; pass-1 order psi-decoupling -> traced PINGPONG redone over all four
GEMMs -> MUFU-poly, a lever failing only after two designed variants; cheap identical-output rounds (H3 glue copies,
forward k-scale prefetch) interleaved; a four-floor speed-of-light model with calibrated peaks; review findings block
only if verdict-changing, at most one tooling commit per round; gate.py stack growth is a warning, ptxas spill /
serialization stays fatal; no CPU work or rebuilds in GPU jobs; waits on job IDs only.

## Round 11: MUFU-POLY variant 1 (D128 pass 1, named numerics candidate): stage 1 slower (lever not yet failed)

- Change (`results/r11/mufupoly_v1/mufupoly_v1.patch`, hd128 P1-R only; D256 code unchanged): the argument and F1
  clamp as shipped; odd columns (half of each thread's elements) computed by a degree-3 minimax 2^f polynomial on the
  FMA pipe with exponent-bit insertion (x clamped to >= -127; -inf -> a denormal that FTZ treats as 0); even columns
  keep MUFU.EX2. Emitted (target functions): MUFU.EX2 128 -> 64, FFMA 768 -> 960, FRND / F2I 0 -> 128; REG 168,
  STACK 8 / 0 unchanged, LOCAL 0, no spill / serialization diagnostics.
- Stage 1 (job 21167357, `gate.py --mode timing --target pass1` vs PSIPF, merged grid): D128 pass 1 265.271 ->
  280.097 ms (**+5.59%, slower**), 37 of 39 D128 shapes slower beyond spread; D256 pass 1 +0.03%; bench.py D128
  matched / stale fwd+bwd +2.5% / +2.8% (the new structured bench predicate flags it).
- Reading: halving the MUFU work moved ~3 extra FMA-pipe / conversion instructions per element onto the issue slots,
  which (with the vector work already dominating the D128 pass-1 step) costs more than the MUFU pipe saved. Not a
  failed lever yet (Hao: two designed variants); a second variant would need to cut issue slots, e.g. a smaller
  replaced fraction or the conversion folded into the existing scale FMA. Counter: D128 pass 1 stays 0.
- Tooling (the round's one bounded commit): gate.py per-symbol stack growth now a warning (A/B still runs); build-log
  spill / C7510 / serialization diagnostics fatal; `--lib` / `--build-log` reuse a prebuilt library (no rebuild in
  the gate); CPU tooling tests opt-in (`--tooling-tests`), run on the cpu partition instead.

Next (Hao's order): the stacked D128 pass-1 candidate H1a + H1b (+ AMAX-REDUX) on the current PSIPF base.

## Round 12: stacked H1a + H1b (D128 pass 1): ACCEPTED (1ffffe2); scope narrowed to D128 causal matched

- Candidate (Hao's first stacked candidate; `results/r12/stack_h1a_h1b.patch`): H1b reapplied as is; H1a ported by
  hand onto it (the P~ conversion, the dV window rescale, the dOt wait and the dV gemm moved to right after the dO
  release, the fp32 P fragment's last reader, before the cross-warpgroup psi barrier). Emitted: the dV QGMMAs
  interleave with the psi-amax shuffles before the 256-thread psi BAR.SYNC; FIFO retire unchanged; resources and
  instruction counts identical to PSIPF; no diagnostics. AMAX-REDUX not included (not ready).
- Gate (job 21168017, full, prebuilt library, vs PSIPF, merged grid): tooling tests 83 passed (beside the build);
  pytest 49 / 3; **bitwise on all four routes vs the baseline and PSIPF**; D128 pass 1 267.181 -> 256.414 ms
  (**-4.03%**), no shape slower beyond spread; bench D128 matched / stale fwd+bwd -6.0% / -5.8%.
- Confirmation (job 21168255, same candidate 2fb59676beb1 vs PSIPF e1119aaaa89c): **-3.99%**, all predicates pass;
  bench D128 -3.0% / -3.4%. On the D128 causal-matched subgrid (scope below): -4.23% / -4.39%, no shape slower.
- Verdict: ACCEPTED. Accepted build 2fb59676beb1 (`.local/frozen/r12_stack`, stamp code_sha 17178b5b6bca). Stack
  ledger: H1a (round 1, -1.27%, one job) and H1b (round 5, -1.64%, one job) were single-job sub-3% results; the
  combination was measured on its own (two jobs) and not inferred from their sum. D128 pass-1 counter stays 0.
- Tooling (the round's bounded commit 656782f): gate.py `--lib` never deletes or self-links its input (B9);
  diagnostics parse C7510 / serialization and nonzero spill counts (B10); fixture fix (Q15).

### Scope change (Hao, during round 12)

Only the matched-delta method counts: target grid D128 causal matched (13 configurations, `--grid d128m`); stale /
full / D256 stay correct but are not measured; re-rank the matched fwd+bwd kernels (forward, producers / glue, pass 0,
pass 1, pass 2); pass 0 gets a place in the order; acceptance as before on this grid only. Recorded in
`docs/kda-sm90/rule-change-r10.md`; implemented in `kernels.py` (grid d128m), `ab.py` (`is_tuned`, decision gated on
the tuned configurations only) and `gate.py` (A/B on d128m, bench matched D128 only).

Ranking over the D128 causal matched configurations (accepted stack arm of the confirmation,
`results/r12/stack_2/ab_all/B_median.json`):

D128 causal matched configurations: 13 (old two-floor model; the four-floor model M3 is pending)

| kernel | measured ms | share of kernel time | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|---:|
| pass1 | 63.460 | 46.0% | 17.748 | 45.712 | 28.0% |
| fwd | 26.423 | 19.2% | 8.761 | 17.662 | 33.2% |
| pass0 | 21.749 | 15.8% | 8.761 | 12.988 | 40.3% |
| glue | 7.505 | 5.4% | 2.631 | 4.875 | 35.0% |
| pass2 | 13.982 | 10.1% | 11.483 | 2.498 | 82.1% |
| prod_fwd | 2.753 | 2.0% | 1.954 | 0.799 | 71.0% |
| prod_bwd | 0.969 | 0.7% | 0.683 | 0.286 | 70.5% |
| twin | 1.048 | 0.8% | 0.910 | 0.138 | 86.8% |
Order from here (waste, the audit direction and the scope): pass 1 (psi decoupling next, after the Q2 / Q3 numerics
gate it needs), with cheap identical-output rounds interleaved (forward k-scale prefetch; H3 glue copies), pass 0
profiled and placed after forward (12.99 ms waste, 15.8% of matched kernel time), then pass 2 / producers.
Counters (D128 matched): pass 1 0, forward 0, pass 0 0, glue 0, pass 2 0.

## Sequence-length focus (Hao, during round 13)

Target grid: D128 causal matched at N 4096 / 8192 in four head / GQA variants (H/Hk 16/16, 16/4, 32/8, 8/8; B for
~32k tokens per call) = 8 target configurations; ranking and acceptance (3% on the target kernel, stacking, second
job) on it; the same variants at 16k / 32k are a no-regression guard; 2k stays correct only. Recorded in
`docs/kda-sm90/rule-change-r10.md`; implemented in `kernels.py` (`D128_TARGET`, `D128_GUARD`, `--grid d128m` = 16
configurations) and `ab.py` (`is_tuned`, `is_guarded`, `decision()`).

## Round 13: the numerics gate (Q2 / Q3) built and self-certified on the accepted build; a pre-existing NaN defect found

- C++ (host code only): `bwd_fp8_core` hands the realized dS' / psi to invocation-owned outputs (the process-global
  capture pointers are gone); a DBG-build op `bwd_fp8_t_dbg` takes the caller's Q / dO twins (the production
  `bwd_fp8_t` signature) and returns that launch's dq / dk / dv, matched row / absolute-row partials, checksums, dS'
  and psi. Shipped (non-DBG) library: all 22 kernel functions byte-identical to the accepted build (job 21168827).
- `benchmarks/numerics.py`: `capture` (all 72 cases of a route, no length / kind limit; fp32 reference with TF32 off;
  relative-L2 errors of O / dQ / dK / dV in the common layout, output and reference identities; with `--dbg` every
  backward group call of the public API is routed to `bwd_fp8_t_dbg`, so the certified gradients come from the launch
  whose workspaces are captured: visited-tile dS' / psi validation and hashes, matched row residual) and `check`
  (complete coverage, equal reference identities, no per-case / output error increase vs base and previous,
  deterministic independent repeat, DBG == production outputs, workspace determinism, `--ws same` equality, matched
  residual within bound). `gate.py --mode numerics` wires the captures and checks per route.
- Residual definition (declared): `|sum_j t_j| / (sum_j |t_j| + 2^8 |delta|)` per row, t_j = P^_j (dP_j - delta) from the
  kernel's partials, reduced in fp64 (n-tile ascending, then warpgroup); bound `sk * 2^-23`. The plain `|sum t| / sum |t|`
  is exactly 1 for any row with a single nonzero term (causal row 0), whatever the rounding residue.
- Baseline self-certificate (jobs 21168827 / 21169100; base = frozen original baseline, prev = cand = accepted stack,
  DBG = the DBG build of the same source, independent repeats): **PASS on default / kv / head / streams2**; worst matched
  residual 5.64e-5 against a bound of 1.22e-4 (d128_scale10). Evidence: `results/r13/selfcert/`.
- **Pre-existing defect found** (present in the frozen original baseline and in every accepted build): all-zero causal
  inputs give NaN forward outputs (522240 of 524288 O elements in a B1 H4 N1024 D128 probe; full attention is clean;
  `results/r13/nan_probe.txt`). The gate lists these cases as baseline defects (the candidate must reproduce the
  baseline's non-finite pattern there). The cause is not yet established; it needs its own fix round (likely a zero
  scale multiplying the masked -inf scores in the forward descale).
- Tooling repairs in the same commit: B11 (actual ptxas spill warnings and any positive spill count fail the gate), Q16
  (bench.py `--modes`, the gate times matched only), Q17 (gate documentation). CPU suite 103 passed (beside the job).

## Round 14: H3, producer Q / K / V copy removal (glue): ACCEPTED

- Change (`results/r14/h3.patch`, `delta_matching/producers.py` only; CUDA library unchanged): strided-input copies of
  the frozen quantizer (`quant._quant_kernel` modes 0 / 1 / 2) and of `_kmean_partial_kernel` read the caller's
  (B, H, N, D) tensors directly (bh -> b = bh // H, h = bh % H; offset b*sB + h*sH + n*sN + d*sD); output codes /
  scales / twins stay contiguous; every arithmetic op identical. The first use of each (kind, shape, strides) signature
  is checked bitwise against the contiguous route (fallback on a mismatch); `DM_STRIDED_PRODUCERS=0` restores the
  copying route. Triton resources (`results/r14/h3_triton_resources.json`): quant MODE0 92 vs 88 registers, MODE1+T8
  62 vs 61, k-mean partial 29 vs 31, no spills.
- Gate (job 21169539, full, prebuilt library reuse, vs `.local/frozen/r13_prod`, target glue on the 8 target
  configurations): pytest 49 / 3 in-process verified; **bitwise on all four routes vs the frozen baseline and the
  previous accepted revision**; glue 5.154 -> 1.662 ms (**-67.75%**), no target / guard shape slower beyond spread;
  bench D128 matched fwd+bwd -2.6%. (Job 21169464 stopped at pytest: the `--lib` path did not install the library's
  build stamp into the worktree, so `kernel_version()` lacked `library`; fixed in gate.py, then rerun.)
- Confirmation (job 21169682, same candidate package / library): glue 5.138 -> 1.657 ms (**-67.75%**), all
  predicates; bench D128 -2.1%. Whole d128m grid: glue -47.1%, prod_fwd +2.0%, e2e fwd -10.0%, e2e fwd+bwd -2.0%.
- Verdict: ACCEPTED. Glue counter stays 0.

Target-grid ranking after H3 (accepted arm of the confirmation, `results/r14/h3_3/ab_all/B_median.json`):

Target configurations: 8 (D128 causal matched, N 4096 / 8192; old two-floor model, the four-floor model is pending)

| kernel | measured ms | share | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|---:|
| pass1 | 26.271 | 45.5% | 7.207 | 19.064 | 27.4% |
| fwd | 11.196 | 19.4% | 3.603 | 7.592 | 32.2% |
| pass0 | 9.165 | 15.9% | 3.603 | 5.561 | 39.3% |
| pass2 | 6.004 | 10.4% | 4.870 | 1.134 | 81.1% |
| prod_fwd | 1.908 | 3.3% | 1.364 | 0.544 | 71.5% |
| prod_bwd | 0.754 | 1.3% | 0.541 | 0.213 | 71.7% |
| twin | 0.828 | 1.4% | 0.721 | 0.107 | 87.1% |
| glue | 1.657 | 2.9% | 2.043 | -0.386 | 123.3% |

## Round 15: the four-floor speed-of-light model (sol4-v1) and the target re-rank

Tooling only (Hao's audit direction 5); no kernel change. `benchmarks/calibrate.py` measures same-allocation peaks;
`benchmarks/sol4.py` takes per kernel the max of four floors: tensor (tile-granular FLOPs / calibrated fp8 GEMM peak),
MUFU (executed MUFU.EX2 lane-ops / 16 per clock per SM), issue (executed warp instructions / 4 per clock per SM) and
bytes (modeled compulsory bytes / calibrated copy bandwidth). Executed counts come from the steady-state SASS loop of
each D128 causal kernel (`results/r15/sol4_counts.json`: fwd 1288 instr / 146 MUFU / 12 QGMMA per step; pass 1 1026 /
32 / 12; pass 0 720 / 64 / 16 over 2 steps; pass 2 325 / 0 / 8) times the declared tile / warpgroup / causal-visited
multiplicities. Triton producers, twins and glue get the bytes floor only (issue not modeled). The glue ledger now
follows the producer mode (strided after H3), which removes the copied-traffic fiction (glue was 123% of its old floor).

Calibration (job 21170192, pool0-01834): fp8 GEMM 1386.1 TFLOPS, copy 3026.3 GB/s, SM clock 1980 MHz, sha256
78a0f4225064. Accepted build r14_h3, d128m grid, 3 repetitions, medians (`results/r15/sol4_accepted.md`).

Target (D128 causal matched, N 4096 / 8192, 8 configurations):

| kernel | measured ms | floor ms | controlling | % of floor | waste ms |
|---|---:|---:|---|---:|---:|
| pass1 | 26.042 | 14.182 | issue | 54.5% | 11.860 |
| fwd | 11.106 | 5.578 | tensor | 50.2% | 5.528 |
| pass0 | 9.083 | 5.466 | tensor | 60.2% | 3.617 |
| glue | 1.653 | 0.665 | bytes | 40.2% | 0.988 |
| pass2 | 5.967 | 5.391 | bytes | 90.3% | 0.576 |
| prod_fwd | 1.902 | 1.510 | bytes | 79.4% | 0.392 |
| prod_bwd | 0.751 | 0.599 | bytes | 79.7% | 0.152 |
| twin | 0.828 | 0.798 | bytes | 96.4% | 0.030 |

The guard (16k / 32k) ranks the same (pass 1 55.1%, fwd 51.7%, pass 0 62.6%, pass 2 92.2%); glue there is 10.9% of its
floor, but all of it sits in the GQA configurations (H != Hk) and is noisy at 16k / 32k (0.27-2.90 ms per call).

Findings:
- Pass 1 is issue-bound, not tensor-bound: 1026 warp instructions per 64x64 step against 12 QGMMA; its issue floor
  (14.2 ms) exceeds its tensor floor (10.9 ms). Levers that cut instructions (MUFU-poly is the wrong axis: MUFU floor
  3.5 ms) come first: the psi / dS' conversion and packing path, and PINGPONG to hide the rest.
- The forward is 50% of its tensor floor and 1.276x FA3's fp8 forward kernel summed over the 8 targets (11.106 vs
  8.706 ms; per config 1.17-1.37x). Its issue floor (4.5 ms) is close to the tensor floor: 1288 instructions per step
  (146 MUFU) leave little issue slack for overlap.
- Pass 0 is 60% of its tensor floor; pass 2 is at 90% of its bytes floor (closed for practical purposes).
- New ranking by target waste: pass 1 (11.9 ms) > forward (5.5) > pass 0 (3.6) > glue (1.0, GQA expansions) > pass 2
  (0.6) > producers (0.5).

Also in this round's tooling commit: B16 (`--lib` stamp: no SameFileError for a package-resident library; a stale
stamp is removed when the reused library has none); `gate.py --mode numerics` never accepts and `numerics.py check`
marks its PASS `ac23_eligible: false` until Hao settles the residual contract. CPU tests: 127 passed (job 21170192).

## Focus change (Hao, during round 16): the training shapes of the paper's 1.67B and of Qwen3-30B-A3B

Four messages during round 16; the last one wins (docs/kda-sm90/rule-change-r10.md): the acceptance set is D128 causal
matched B2 N8192 (pretraining) and B1 N65536 (the 64K context extension), each at Hq 16 / Hk 4 (the paper's 1.67B
GDN/GQA hybrid, arXiv 2609.37852) and Hq 32 / Hk 4 (Qwen3-30B-A3B): four shapes, equal weight per shape (geometric mean
of per-shape ratios). Rules otherwise unchanged. Everything else only stays correct. The intermediate sets (16/4 at
B2 8k / B4 4k / B1 64k; plus 32/4; six shapes at 16k tokens; eight shapes) were superseded within the round. New
report: `results/paper_c2.md` (Appendix C.2 protocol). Next lever named by Hao: per-warpgroup psi for pass 1.

## Round 16: KSPF (forward k-scale prefetch): REJECTED; sol4-v2; paper protocol report

- KSPF variant 1 (`results/r16/kspf/kspf_v1.patch`): in the D128 PV-first forward step the current tile's four k8s
  values are loaded at the top of the step (before the V wait) instead of after the QK gemm's wait (accepted SASS:
  4 LDG at 0xe040 right after `WARPGROUP.DEPBAR`, consumed by the descale). Identical arithmetic.
- Gate job 21170520 (full, vs r14_h3, old d128m grid): bitwise on default / kv / head / streams2 vs the frozen baseline
  and the previous accepted revision; REG 168 unchanged, no diagnostics; forward -0.38% over the old 8 targets
  (guard -0.71%) -> fails 3%.
- On the round-16 acceptance set (job 21170691, ab.py, 3 reps): forward -0.51% (geometric mean; per shape 0.997 /
  1.000 / 0.989 / 0.993; sum -0.47%), no shape slower beyond spread -> REJECT. Reading: the other consumer warpgroup
  already hides the k-scale load latency; variant 2 (load one step ahead) cannot recover more than the measured ~0.5%,
  so it is not built now (the lever has one designed variant; forward counter unchanged).
- sol4-v2 (round-15 review B17-B20): issue / MUFU floors from a control-flow ledger (cheapest path through each step
  loop doing the step's full QGMMA work; rare and mutually exclusive arms never summed; all MUFU opcodes, predicated
  MUFU excluded; peeled steps and prologue / epilogue / producer work excluded: loop-only LOWER bounds); inputs bound
  (calibration digest recomputed, run calibration id, counts version / declarations / library hash, the exact grid);
  glue ledger from the route actually taken (producer backend from kernel names, the backward head split with its
  real copies: per-head slices at B1 are views, only the output concatenation moves bytes); FA3 timings are per-field
  medians over the same repetitions (ab.median_records). Per step and warp (accepted library 756158cb1226): fwd >= 1284
  instructions (static 1288), pass 1 >= 656 (static 1026), pass 0 >= 280.5 per step (static 720 per 2 steps), pass 2 >= 68
  (static 325; a data-dependent zero-tile skip path exists, charged as full work like the tensor floor).
- Round-15 data re-modeled (results/r16/sol4v2_r15data.md): pass 1 is TENSOR-floor-controlled at 42.0% (the v1 "issue
  bound at 54.5%" claim is withdrawn: its issue floor counted the promotion / rescale arms every step); forward 50.2%,
  pass 0 60.2%, pass 2 90.3%; forward / FA3 fp8 forward = 1.2628 (v1's 1.276 mixed FA3 repetition 0 with medians).
- The accepted build on the acceptance set (calibration job 21170691: 1388.1 TFLOPS fp8, 3026.0 GB/s, sha 827e29e9ebcc;
  `results/r16/sol4v2_accepted.md`):

| kernel | measured ms | floor ms | controlling | % of floor | waste ms |
|---|---:|---:|---|---:|---:|
| pass1 | 188.484 | 78.603 | tensor | 41.7% | 109.881 |
| fwd | 76.269 | 39.394 | tensor | 51.7% | 36.874 |
| pass0 | 63.701 | 39.302 | tensor | 61.7% | 24.400 |
| pass2 | 37.703 | 35.589 | bytes | 94.4% | 2.114 |
| glue | 3.172 | 2.373 | bytes | 74.8% | 0.799 |
| prod_fwd | 1.149 | 0.752 | bytes | 65.5% | 0.397 |

  Forward / FA3 fp8 forward kernel over the set: 1.311. Ranking by waste: pass 1 >> forward > pass 0 >> pass 2.
- Paper protocol (`results/paper_c2.md`, job 21170691, 5 interleaved rounds): B2 N8192 16/4: accepted build fwd 506.7,
  bwd 458.9, fwd+bwd 471.6 TFLOP/s (original baseline 518.0 / 443.4 / 462.4; FA3 BF16 696.4 / 506.6 / 549.2; cuDNN BF16
  582.6 / 475.3 / 501.7; TE FP8 not run: transformer_engine absent). The paper reports 661.0 / 421.5 / 470.2: our
  backward and fwd+bwd match or exceed it, our forward is 23% below the paper's (FA3 BF16 here is 1.17x the paper's
  matched fwd+bwd, so this H100's FA3 reference also differs from the paper's ratio 1.80x). 32/4: accepted 564.4 / 467.7
  / 491.7 (baseline 563.6 / 451.9 / 478.9; FA3 BF16 723.7 / 521.2 / 566.5; cuDNN BF16 616.2 / 470.1 / 504.2).

## Round 17: residual contract implemented (v2 -> v3), breadth search, pass-0 C2 + C3 ACCEPTED (helper candidates)

Hao during round 17: the residual decision (residual-v2), then residual-v3 after the self-certificate; the goal / stop
rule (every main kernel >= 90% of its calibrated sol4 floor on the four training shapes; first milestone forward >=
FA3 fp8 forward); breadth search with parallel cheap screens; helper agents (kda-pass1 / kda-fwd / kda-pass0) post
candidates on their branches and this loop gates / integrates them; a progress A/B vs the original after every
accepted round; paper C.2 reconciliation. All recorded in docs/kda-sm90/rule-change-r10.md.

- Numerics: residual-v2 (per-row bound k 2^-23, degenerate exemption, per-row non-regression; hashed per-row sidecars;
  identity hardening) implemented; the accepted build's self-certificate failed only the absolute bound (10204 /
  334592 rows on all four routes; |delta|-conditioned variant 9667) -> Hao dropped it: residual-v3 = per-row
  non-regression at round-off + no exemption growth + deterministic + error not larger + named change. Accepted build
  r17_c3 re-certified under v3 on all four routes (PASS, 4 inherited NaN defect cases, 8378 exempt rows, jobs
  21174137-41, results/r17/cert_v3). OPEN (for Hao): the "fp32 error not larger" clause has zero tolerance and rejects
  any output change at noise level: the forward helper's O-only candidate (lazy O rescale + rho precompute) has 0
  residual regressions and an unchanged exemption set, but 22 of 288 case/output errors go up and 22 go down by
  ~1e-6 relative (max 5.7e-6); the pass-0 helper sees the same (+7e-9 .. +1.3e-7) and near-one-hot rows (residual ~1
  already) moving by up to 0.18 under any delta-rounding change.
- Breadth (results/screens.md): forward worker, 18 variants (13 bitwise candidates, 5 probes): no bitwise winner (best
  RHOPRE -0.9%, noise; KSPF v2 -0.6%); probes: rho rescales ~9%, S descale ~5% of the forward, O *= alpha ~1%,
  removing all steady PV work only -17..19%; every barrier alternative +1..12% -> the forward's remaining levers are
  numerics changes (the forward helper's FWDLAZY / FWDDFOLD candidates). Pass-1, pass-0 and per-warpgroup-psi workers
  were still screening at the end of the round (their logs land in .local/screens/*_log.md).
- C2 (pass 0 P0RTR transposed quad reduce + P0SPF 8-byte scale loads, kda-pass0 helper): gate job 21172404 bitwise on
  all four routes, pass 0 -5.21% geomean (0.948 / 0.950 / 0.947 / 0.946), confirmation 21172600 -5.27%: ACCEPTED
  (5ff8956, 45aa160).
- C3 (pass 0 P0LPT global longest-first dispatch + P0W4 four consumer warpgroups per 256-row CTA + template WG count
  with 2-WG fallback): gate job 21173322 (--allow-new-functions) bitwise on all four routes, pass 0 -7.67% vs C2
  (0.885 / 0.955 / 0.899 / 0.956), confirmation 21173891 -7.64%: ACCEPTED (499abb9, 97884e6, 963771f).
- Accepted build r17_c3, % of calibrated limit (sol4-v2, job 21174139, calibration 641cda28f8c0;
  results/r17/accepted_c3/sol4v2.md): pass1 40.1%, forward 49.7%, pass0 65.9%, pass2 94.7% (glue 74.9%); forward /
  FA3 fp8 forward 1.319. (After C2, job 21172815: pass1 39.8, fwd 49.0, pass0 62.0, pass2 94.3.)
- Progress vs the original 3d4cd5b on the four shapes (results/progress): e2e fwd -2.37%, fwd+bwd -5.10%.
- Paper C.2 (B2 N8192): 16/4 fwd+bwd 491.9 TFLOP/s (paper 470.2; original 465.4; FA3 BF16 545.7; cuDNN BF16 511.9);
  forward gap reconciled (kernel 704 TFLOP/s; the per-call window adds ~0.3 ms host work); the paper's FA3 BF16 ratio
  does not reproduce on this H100 (results/paper_c2.md).
- Waiting at the gate: C4 (P0ALL, split.py: pass 0 once on all heads; needs the equivalence schema for one all-heads
  delta call on split routes), forward C1 / C2 (numerics, need Hao's error-tolerance decision).

Evidence for the open tolerance decision (kda-pass0 helper, residual-v3 checker on the default route vs the r17_c3
certificate; its reports in kda-work/p0/num): output-changing pass-0 cuts on the C3(+C4) stack, determinism and exempt
sets unchanged in all three:
- DEFER (fp64 per-lane row partials; pass 0 -16.6% vs -14.4% for the stack): 77 per-row regressions, ALL on stress
  cases (scale3 37, scale10 9, outliers 31; zero on the Gaussian cases); fp32 error +0.1 .. +3 ppm.
- LAZY (dP descale per 8-key half partial; -18.3%): 250 regressions (Gaussian cases 1-3 each, stress 242); <= +2.0 ppm.
- LAZY + DEFER (-21.25%): 283 regressions (same pattern); <= +1.7 ppm.
The stress-case rows already sit near residual 1 in the accepted build. A rule with a noise tolerance on the fp32 error
(<= ~1e-5 relative) and an ill-conditioned exemption for rows whose ACCEPTED residual is already above a threshold
(e.g. 1e-3) would accept DEFER and the forward C1; LAZY would still regress a handful of Gaussian rows.
- S17-p1-39 (pass 1: early dV window rescale + deferred double-buffered dS' stores, from the round-17 pass-1 screen):
  gate job 21174913 bitwise on all four routes vs the baseline and r17_c3, pass 1 -4.68% geomean, bench -1.9%;
  confirmation 21175475 -4.87% (same library be70f302): ACCEPTED (accepted root r17_p1).
- Per-warpgroup psi (Hao's named lever; S17-psi-01..09, results/screens/psi_log.md): implemented completely (pass 1
  WG-local amax, psi[..., 2], pass 2 two accumulators with per-half windows, DBG op), numerically sound (errors: 34
  better, 108 equal, 2 worse by +0.03% / +0.04%), but pass 1 -1.2..-1.8% and pass 2 +7.4..+8.6% (the second
  accumulator's per-tile rescale): pass1 + pass2 +0.1..+0.8% -> REJECTED (variants 1-9; lever: 2+ designed variants
  failed; the round-2 -4.2..-4.8% predates H1a / H1b, which already hid most of the psi barrier).
- Pass-0 worker (S17-p0-01..23, results/screens/pass0_log.md, screened vs r14_h3): best bitwise S17-p0-20 (reduce-
  scatter + nomaskloop + skipdead + setmaxnreg 240/24) -6% vs r14_h3, but its reduce-scatter equals the accepted P0RTR
  and C3 restructured the kernel: the nomaskloop / skipdead / register-split pieces need a port onto r17_c3. Profile:
  pass 0 is issue-bound with 192 fixed instructions per 64-key block for bitwise outputs -> bitwise ceiling ~70-75% of
  its floor; 90% needs a numerics change (fold the +8, lazy dP descale), cf. the pass-0 helper's DEFER / LAZY.
- Accepted build r17_p1 (job 21175599, calibration in results/r17/accepted_p1): % of calibrated limit pass1 45.1%,
  forward 52.4%, pass0 69.8%, pass2 94.7%; original -> now e2e fwd+bwd -7.01%; paper C.2 16/4 fwd+bwd 503.7 TFLOP/s.
  Note: the calibrated fp8 GEMM peak varies ~6% between jobs / nodes (tensor floors 75.3 -> 80.0 ms for the same work),
  which moves every percentage with it; per-round comparisons should use the same-job calibration as here.
- residual-v4 implemented (rule-change-r10.md); numerics gates of forward C1, pass-0 DEFER and LAZY submitted
  (jobs 21176380-82) against r17_p1 with its DBG pair.
- CORRECTION to commit 23fe1e7: the "DEFER / LAZY pass the numerics gate, no gain" result is void. The patch files
  (kda-work/p0/tools/{defer,lazy}.patch) only add code under #ifdef KDA_P0DEFER / KDA_P0LAZY and I built them without
  the defines, so jobs 21176380 / 21176381 gated the unchanged kernel. The real candidates are commits on
  origin/kda-sm90-pass0-num (e3275a3 P0DEFER, c49c0d6 P0LAZY on it); c49c0d6 is in the numerics gate (job 21177096).
- Forward C3 (kda-fwd helper: BM192 = 192 x 128 tiles, three consumer warpgroups, on FWDLAZY + FWDDFOLD + RHOPRE;
  -12.9% / -13.2% forward in the helper's two A/B jobs): FAILS residual-v4 on the default route (job 21176828): the
  inherited all-zero causal NaN defect grew (O n_nan 522752 vs 522240), 70 well-conditioned stress-case rows regress
  beyond round-off (scale3 36, outliers 18, scale10 15, mqa 1), and the per-output means rise by ~3e-7 -> REJECTED as
  submitted; returned to the helper.
- Pass-0 LAZY + DEFER (c49c0d6, the real commit; -7.50% pass 0 in the helper's A/B): FAILS residual-v4 (job 21177096):
  well-conditioned rows regress beyond round-off in Gaussian cases too (gqa4 3 rows, worst 2.36e-5 -> 4.43e-5; mqa 2;
  gqa3_promote 2; heldout 1) and in stress cases (scale3 25, outliers 20, scale10 4) -> REJECTED as submitted.
  Reading: any change to the rounding of pass 0's delta moves well-conditioned rows by more than k 2^-23; per-row
  non-regression keeps pass 0's numerics effectively frozen unless a change makes delta closer to pass 1's sum row
  by row. Bitwise pass-0 levers (the pass-0 worker's nomaskloop / skipdead / register split, ported onto r17_p1) are
  the next pass-0 candidates.
- Forward C3f (3565487: BM192 + FWDLAZY + FWDDFOLD + RHOPRE + 3 K/V stages + FLT_MIN floor; -12.8% forward in the
  helper's A/B) and C3n (b1613d8: the same without FWDDFOLD; -5.82% / -5.88% in two helper jobs) re-gated with the
  recorded decisions (LSE +-1 ulp envelope from my own r17_p1 control builds, inherited-defect repair, mean-error noise
  1e-6; jobs 21177735 / 21177736): C3f FAILS on one d128_mqa row beyond accepted + round-off + envelope; C3n FAILS
  only because its zero-input NaN pattern moved (the zero-scale floor is on the fold path only). Requested C3n2 with a
  zero-scale guard on the non-fold path. Pass-0 helper: FOLD (-13.1%) and the identity-exact NORM also fail v4; it
  concludes the pass-0 numerics path is closed under strict per-row non-regression and returns to bitwise levers.
- Forward C3n2 (C3n + zero-scale floor on the non-fold path): numerics gate 21177865 PASS on all four routes (with the
  LSE envelope and defect repair), forward -8.49%; confirmation 21178214 -8.66%. Superseded by C3g before merging.
- Forward C3g (kda-fwd helper; BM192 = 192 x 128 tiles with three consumer warpgroups, FWDLAZY lazy O rescale,
  FWDDFOLD S-descale fold on UNMASKED steps only, RHOPRE, 3 K/V stages, zero-scale floors; NAMED NUMERICS CHANGE):
  numerics gate 21178215 PASS on default / kv / head / streams2 (residual-v4 with the LSE +-1 ulp envelope, the
  repaired zero-input defect: finite O = 0), forward -12.89% geomean (0.873 / 0.867 / 0.876 / 0.868), bench -2.8%;
  confirmation 21178561 (same library e21133d5e72f) -12.68%: ACCEPTED (cherry-picked c6ebf3c .. b0db717 code commits;
  accepted root r17_fwd + r17_fwd_dbg + LSE controls). sol4 declares the forward as BM192 (>= 599 instructions, >= 75
  MUFU per warp per 64 x 128 step).
- Stacking candidates held (bitwise, >= 1% in two jobs, < 3% alone): pass-1 C2 SHFL8 (-1.78% / -1.50%), pass-0 C5
  nomaskloop + register split (-1.27% / -1.42%). 64K split route (Hao via kda-fuse): forced per-KV-group splits and
  workspace limits 40 / 72 GB change the 64K fwd+bwd by <= +-0.5% (job 21177857; glue 0.5-3 ms of 115-230 ms);
  in-place outputs (~1%) need an out= path in the ops (queued).
- Accepted build r17_fwd (job 21178641; results/r17/accepted_fwd/sol4v2.md): % of calibrated limit pass1 43.9%,
  forward 59.0%, pass0 68.0%, pass2 94.5%; forward / FA3 fp8 forward kernel 1.163 (was 1.32); original -> now e2e fwd
  -13.59%, fwd+bwd -9.19%; paper C.2 16/4 fwd 581.6 / fwd+bwd 502.4 TFLOP/s, 32/4 fwd+bwd 523.1 (cuDNN BF16 500.0).
- Pass-1 helper on r17_fwd: no bitwise stack of its own reaches 3%; best SHFL8 + QST=3 / DOST=3 at -2.15% / -2.45%
  (SHFL8 alone -2.20% / -1.74%), held for stacking with a >= 1% bitwise partner; ~135 screened pass-1 variants
  (DVZI, LSEPRE, PV4, FENCE2, PCONVEARLY, EXPSPLIT, STLATE, REDUX trees, EARLY / KRS / DKSPLIT) neutral or slower
  (CANDIDATES.md on origin/kda-sm90-pass1-v2).
- Forward helper C4 (PRETURN: lazy rebase / rho_1 / promote weight computed before the ring sync; bitwise, O sha equal
  on the four shapes): forward -1.10% / -1.12% vs r17_fwd in two jobs -> held for stacking (needs a >= 3% forward
  stack); C4b (+ K try-wait token) -0.58% / -0.75%, rejected.
- Pass-0 stack (kda-pass0 helper: C5 = P0NML mask-free leading-stage loop for W4 + consumer register split 104 / 24,
  ported from my pass-0 screen; C4 = P0ALL, split.py runs pass 0 once on all heads at D128 matched): equivalence
  manifest v4 adopted (853f7e6: canonical all-heads matched deltas); gate 21179675 (byte references = the r17_fwd v4
  dumps, see rule-change) bitwise on all four routes, pass 0 -3.19% geomean (0.993 / 0.945 / 0.985 / 0.951, the gain
  is mostly the 64K split route), bench +0.2%; confirmation 21179800 (library 11a7eebdcabe) -3.43%: ACCEPTED (accepted
  root r17_p0s).
- Forward stack C4 (PRETURN) + kda-fwdport SMSC (k/v-scale smem staging): -2.46% / -2.42% vs r17_fwd, below 3%, not
  yet gated.
- Pass-0 helper status after the stack: every bitwise pass-0 lever measured with >= 2 variants and exhausted
  (CANDIDATES.md on origin/kda-sm90-pass0); the W4 kernel's remaining loss is tensor / vector overlap across
  warpgroups (one S / dP pair per warpgroup fits the register budget; cross-iteration pipelining is serialized by
  ptxas C7517; more accumulators spill); the output-changing cuts (FOLD -13%, LAZY, DEFER, NORM) fail residual-v4.
  Next structural lever (proposed): share pass 0's S / dP work with pass 1 (pass 0 recomputes the same products pass 1
  needs, e.g. on the diagonal tiles), i.e. a cross-kernel redesign rather than further pass-0 tuning.
- Accepted build r17_p0s (job 21179953): % of calibrated limit pass1 42.7%, forward 58.2%, pass0 70.6%, pass2 94.3%;
  forward / FA3 fp8 1.146 (FA3-referenced 87%); original -> now e2e fwd -12.9%, fwd+bwd -8.9% (this job; job-to-job
  calibration / clock spread ~1%); paper C.2 16/4 fwd+bwd 505.7 (cuDNN BF16 505.8), 32/4 528.5 TFLOP/s.
- Forward helper: C4 + SMSC stack stays at -2.46% / -2.42% (not READY); the STACK 16 -> 40 growth is in the
  non-causal instantiation (nothing to recover); two masked-step pieces failed (+18% / +20%, in-step branches spill;
  S52-S53). Pointed it at a third bitwise piece on BM192 and at the ~0.3 ms fwd_fp8 host path (end-to-end lever).
- Forward host-path lever withdrawn: fwd_fp8 host time is 0.015 ms per call (forward helper, job 21180725, nsys);
  the ~0.3 ms in paper C.2's per-call window is idle-GPU launch latency (results/paper_c2.md correction).
- EXPBX64 (kda-p1packed helper; named numerics change: the pass 0 + pass 1 exp line with the 2^8 offset folded into the
  per-row bias off the masked diagonal 64-key range, the certified line exactly on it): numerics gate 21181579 PASS on
  all four routes (residual-v4 vs r17_p0s with its DBG pair and LSE +-1 ulp envelope), pass 0 -4.50% geomean, bench
  -1.4%; confirmation 21181901 (library 5bbf85214faa) -4.04%: ACCEPTED (accepted root r17_pk).
- Pass-1 stack (kda-pass1-c3: SHFL8 + QST 3 / DOST 3, bitwise): full gate 21181578 bitwise on all four routes, pass 1
  -2.26% (< 3%) -> not accepted alone; held for stacking.
- Accepted build r17_pk (job 21181967, calibration 1481.2 TFLOPS = the fixed reference): % of calibrated limit pass1
  41.1%, forward 54.9%, pass0 70.0%, pass2 94.1%; forward / FA3 fp8 1.155; original -> now e2e fwd -13.1%, fwd+bwd
  -11.2%; paper C.2 16/4 fwd+bwd 510.5 (cuDNN BF16 509.6), 32/4 533.3 TFLOP/s. (On the previous job's 1409 TFLOPS
  calibration the same build would read ~5% higher; see the calibration-stability decision.)
- Pass-1 helper on r17_pk: SHFL8 alone (kda-sm90-pass1-c2pk, release flags unchanged) -1.10% / -1.07%, bitwise on all
  four routes (job 21182433) -> the held pass-1 stacking piece; QST3 / DOST3 no longer help after EXPBX64 (+0.25% /
  +0.23%) and are dropped; SHFL8 + QST3/DOST3 -1.18% / -0.50%.
- sol4c adopted (Hao, kda-solcheck evidence; rule-change-r10.md): r17_pk by sol4c (results/r17/accepted_pk/sol4c.md):
  pass1 66.3%, forward 78.5%, pass0 94.0%, pass2 94.1% (pass 0 and pass 2 meet the 90% target; forward and pass 1 do
  not). Forward helper DONE (54 variants, S1-S54; held C4 + SMSC bitwise -2.4%). Pass-0 work stops.
- Pass-1 helper DONE (per-element ledger on r17_pk + SHFL8, CANDIDATES.md on origin/kda-sm90-pass1-v2 f1ec03d): the
  unmasked step executes ~698 instructions per warp-step (993 static); per element 8 (exp 3: FFMA, F1 clamp, MUFU; dS 2;
  amax 1; scale 1; e4m3 conversion 1); per step +32 per-row, 64 FFMA dK fold, 64 FMUL dV rescale on half the steps,
  ~33 promote. Every remaining item is fixed by the numerics contract (F1 clamp, fp32 dK fold), has no sm90 shortcut
  (no 3-input FMNMX, no scaled e4m3 convert; packed variants break even), was measured neutral, or fails residual-v4
  (PW, EXPB, windowed dK). Only SHFL8 paid (-8 SHFL / warp-step, ~-1.1%, held). With the forward and pass-0 helpers
  also DONE, the remaining gaps (forward 78.5%, pass 1 66.3% of sol4c) need structural or numerics-contract changes
  rather than further scheduling / instruction trims within the current kernels.

## Round 18 (started): Hao's structural + numerics direction

- results/REPORT.md written (interim; updated per accepted round). Two workers launched: structural pass-1 redesign
  reusing pass 0's work (S18-st-*), cheaper exp clamp / dK fold as named numerics changes (S18-nm-*), both screened
  against r17_pk (screen.sh now references r17_pk; byte reference .local/equiv4/r17pk_default.pt).
- Forward helper: under residual-v4 the clamp / rho / promote / exp variants it can name save < noise or fail v4 by
  1-3 orders; the remaining 15.5% to FA3 fp8 is the faithful per-group P basis. Directed to build and screen shared-
  basis P codes per 128-key tile (3 WGs; 2 WGs + BN256) as a named numerics change: residual-v4 decides.
- Forward shared-basis P codes (FWDSHB; kda-sm90-fwd-shb dbe37f3, 3 WGs / BN128): forward -4.41% / -4.33% vs r17_pk,
  but residual-v4 FAILS on the default route (82 problems): matched O error rises by 1.3e-3 .. 4.2e-3 in 10 of 23 cases
  (limit 1e-5; e.g. gqa3_promote 4.604e-2 -> 5.025e-2) and 7 matched rows beyond the envelope; the smaller groups'
  codes lose precision on the shared basis (more e4m3 subnormals): a real accuracy loss, REJECTED. 2 WGs + BN256:
  +8.3%. Forward helper back to DONE; the per-group P basis is the forward's remaining cost under the contract.
- kda-pass1 C4 (SHFL8 8-shuffle dS' transpose + the exp clamp min(t, 12) kept only on the certified masked-tile line
  that pass 0 masks (EXPBX64's predicate), dropped on the folded line elsewhere: -32 FMNMX per thread per step in
  pass 1 and per tile in pass 0; a named numerics change, bitwise on all 72 x 4 certification cases): gate 21191514
  (target pass 0 per the two-kernel rule) bitwise on all four routes, pass 0 -3.53%, pass 1 -3.31%, e2e fwd+bwd
  -2.21%, bench -5.7%; confirmation 21191853 (library 13235f129191) pass 0 -3.86%, pass 1 -3.08%, e2e -2.31%:
  ACCEPTED (accepted root r18_c4).
- Structural worker (S18-st-01..08, results/screens/structural_log.md): no candidate. Family B (pass 1 computes the
  matched delta, pass 0 gone): ceiling with pass 0 removed -17.6 / -18.7% backward, but in-pass-1 row sums +13% on pass 1
  and the minimum cross-CTA exchange +100% on pass 1 (+42% backward); infeasible at N65536 (512 key-block CTAs per row
  on 132 SMs). Family A (pass 0 hands its products to pass 1): a bitwise handoff needs 8 B / element (17.2 GB at 8K
  16/4 = 5.7 ms vs a 2.9 ms backward); even a free handoff saves only 5.5-6.4% of the backward; diagonal-only ~0.25%.
  New: a proof that the exp clamp cannot fire when |LSE| < 4096 (the forward's LSE >= the row max): a bitwise
  justification for C4's clamp drop.
- Numerics worker (S18-nm-01..12, results/screens/numerics_p1_log.md): the clamp lever is C4's; exact guarded clamp
  forms are slower (+4..+16%); the dK fold: deferred fold spills (+29%), DKACC (dK in the wgmma accumulator, no fold)
  neutral on r18_c4 (-0.4 / +0.1%) and fails residual-v4 (dQ / dK errors +2e-5..1.2e-3); no cheaper exact fold on sm90.
  With every helper DONE, pass 1 (66% of sol4c) and the forward (78.5%) have no remaining lever within the numerics
  contract that the campaign could find.
- Accepted build r18_c4 (job 21191934, same-job calibration 1454.9 TFLOP/s): sol4c pass1 70.0% (68.8% at the fixed
  1481.2 reference), forward 80.1% (78.6%), pass0 100.2% (98.4%), pass2 94.3%; forward / FA3 fp8 1.162; original ->
  now e2e fwd -14.0%, fwd+bwd -13.1%; paper C.2 16/4 fwd+bwd 521.9 TFLOP/s (cuDNN BF16 510.2), 32/4 548.6.
- Pass-1 helper on r18_c4: GUARDLITE (drop the F2b psi floor and / or the dV ratio clamp) bitwise on all 72 default
  cases but neutral (pass 1 -0.60 .. +0.45%, the guards are off the critical path); the rest of the per-element path
  (exp, dS, amax, scale, e4m3, dK fold) cannot pass residual-v4 (P^ changes move per-row residuals; dS' code / dK
  accumulation changes move per-case errors well past 1e-5). DONE again (CANDIDATES.md on kda-sm90-pass1-v2 8b2e4ff).

## Round 19 (started): own rounds in parallel with the helpers (Hao)

- Workers launched: forward SASS diff vs FA3 fp8 per step and instruction-gap cuts (S19-fwd-*); pass-1 per-element
  instruction cuts on r18_c4 under residual-v4 (S19-p1-*). r18_c4 DBG pair + LSE controls building (job 21192664) as
  the numerics references. kda-p1fresh prototypes will be gated when they land.
- Pass-1 cut worker (S19-p1-01..06 on r18_c4, results/screens/pass1_cuts_log.md): six bitwise variants (DVFRESH
  zero-start dV window -15 instr / warp-step; scale-load / psi-pointer hoists; float4 dV overlay; psi store and
  per-row work moved), none >= 1% in both screens (best -0.33 / +0.02%); moved work +1.3..2% slower; float4 / hoist
  variants grow STACK 0 -> 8. Reading: bitwise-removable instructions sit in gemm-wait shadows; the ~0.05-point
  sensitivity holds only on the S-retire -> dK-issue per-element chain (where SHFL8 / the clamp drop paid). dS in one
  op impossible (two products of P^); dropping the amax pass leaves only a 1e2-1e3 too loose bound; packed f16x2 dS
  saves <= 16 instr (~1.2%) even if v4 passed. No candidate.
- Forward helper round (on C3g + C4 + SMSC): phase probe per WG step ~4020 cycles: pre-turn 9%, waiting for the ring
  token 24%, the ring turn 33% (PV gemms + rescale + promote + QK issue), post-turn 33% (QK wait + softmax): the forward
  is ring-turn-bound, softmax has slack. Two-token rings (bitwise): TOK2 named barriers deadlock (lapping); TOK2b smem
  semaphores +28.9% (vs its 1-token control +48.6%: a second token is worth ~12% but the spin loop spills); TOK2p
  parity-alternating barriers +23.1% (control +19.8%). At 3 WGs any extra per-thread state or branch spills at 160
  registers. No candidate; C4 + SMSC (-2.4%) stays the only held forward piece.
