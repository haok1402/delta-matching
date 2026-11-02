# Pass-1 sparse timeline (round 4)

## Method

- Build: `DM_TRACE=1` (`-DKDA_TRACE=1`, `csrc/kda_prof.h`), timeline-only, never shipped; compiled out by default.
  Sample: CTAs with linear block id lb % 17 == 0 and lb / 17 < 8 (8 CTAs per launch); the first 12 m-block steps of each
  sampled CTA's first tile (covers several stage wraparounds of the 2-stage pipelines). Actors: the first thread of each
  consumer warpgroup and the producer's elected lane. Every (variant, sample, step, actor, event) has its own uint64
  slot (absolute clock64; no atomics) plus a per-sample header (n_block, head, batch, SM id, m_block_min, block id, start
  clock, K/V issue clock). Event ids are listed in `benchmarks/trace_pass1.py`. Timestamps are compared only within one
  CTA (one SM). Each trace is one unsplit API backward (one pass-1 launch) after a warm-up call; the build root is
  selected explicitly (forced-unsplit route, package / library / sha256 validated: Q5).
- Arithmetic: the trace build and the dense-probe build are BITWISE EQUAL to the accepted DO2 build on the full
  72-case captured-intermediate suite (results/r4/equivalence_*.txt).
- Perturbation (pass-1 ms, median of 3 rotating-order repetitions in one job, results/r4/time_*.json): the timeline build
  is +14 .. +16% slower than plain (the sample predicates and extra stack exist in every CTA), the dense build +17 .. +26%.
  How the overhead distributes over phases is unknown; the cycle counts below are of the instrumented kernel.

| shape | plain DO2 | dense probes (DM_PROF) | sparse trace (DM_TRACE) |
|---|---:|---:|---:|
| D128 B1 H16/16 N32768 causal | 15.490 | 18.116 (+17.0%) | 17.695 (+14.2%) |
| D128 B4 H16/4 N8192 causal | 3.876 | 4.654 (+20.1%) | 4.490 (+15.9%) |
| D128 B4 H16/16 N8192 causal | 3.886 | 4.621 (+18.9%) | 4.461 (+14.8%) |
| D128 B4 H16/16 N8192 full | 7.803 | 9.295 (+19.1%) | 9.009 (+15.5%) |
| D256 B1 H8/8 N32768 causal | 16.082 | 19.652 (+22.2%) | 18.440 (+14.7%) |
| D256 B4 H8/8 N8192 causal | 4.002 | 5.046 (+26.1%) | 4.645 (+16.1%) |
| D256 B4 H8/8 N8192 full | 7.966 | 10.013 (+25.7%) | 9.158 (+15.0%) |

- Resources (results/r4/resource_pass1_*.txt): plain DO2 hd128 pass 1 REG 168 / STACK 8 (non-causal), 0 (causal); the
  trace build REG 168 / STACK 16; d256 STACK 0 -> 8 / 32; LOCAL 0 everywhere.

## Timelines (medians over the 80 interior sampled steps per configuration; cycles)

Every quantity is a BRACKETED REGION between two probe events of the instrumented kernel: it contains the named wait
plus whatever pipeline-state, release, scale-prefetch and probe work lies between the two markers. "region_*_wait" are
consumer pipeline-wait regions; "region_*_retire" are warpgroup_wait regions (dV / dK also include the release and
state update that follow); producer regions for dOt / Q / Qt start at the previous producer event (no separate acquire
entry marker); q_issue_lead is how long before the consumers asked for Q(m+1) the producer had ISSUED it (not when it
landed). "vec_regions" are a warpgroup's instrumented vector regions (exp, dS, amax, conversions, transpose /
exchange, barriers, store, fold); "in_*_outstanding" is the part of them inside the other warpgroup's software-
outstanding wgmma windows (issue to the observed wait / release): a software bound, not tensor-pipe activity.

### D128 B4 H16/16 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3849 |
| step_wg1 | 3866 |
| offset | -61 |
| region_Q_wait | 82 |
| region_dO_wait | 80 |
| region_dOt_wait | 62 |
| region_Qt_wait | 85 |
| region_S_retire | 9 |
| region_dP_retire | 9 |
| region_dV_retire_release | 20 |
| region_dK_retire_release_state | 253 |
| region_psi_barrier | 7 |
| vec_regions_wg0 | 2471 |
| vec_regions_wg0_in_wg1_outstanding | 708 |
| vec_regions_wg1 | 2574 |
| vec_regions_wg1_in_wg0_outstanding | 1078 |
| p_region_dO_acquire | 878 |
| p_region_dOt_acquire_bracket | 1613 |
| p_region_Q_acquire_bracket | 148 |
| p_region_Qt_acquire_bracket | 142 |
| q_issue_lead | 3978 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 27%; WG1 in WG0's: 40%.

### D128 B4 H16/16 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3938 |
| step_wg1 | 3944 |
| offset | -49 |
| region_Q_wait | 65 |
| region_dO_wait | 84 |
| region_dOt_wait | 64 |
| region_Qt_wait | 82 |
| region_S_retire | 26 |
| region_dP_retire | 9 |
| region_dV_retire_release | 23 |
| region_dK_retire_release_state | 255 |
| region_psi_barrier | 7 |
| vec_regions_wg0 | 2398 |
| vec_regions_wg0_in_wg1_outstanding | 534 |
| vec_regions_wg1 | 2608 |
| vec_regions_wg1_in_wg0_outstanding | 1022 |
| p_region_dO_acquire | 986 |
| p_region_dOt_acquire_bracket | 1674 |
| p_region_Q_acquire_bracket | 152 |
| p_region_Qt_acquire_bracket | 144 |
| q_issue_lead | 4056 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 22%; WG1 in WG0's: 37%.

### D256 B4 H8/8 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3801 |
| step_wg1 | 3801 |
| offset | -61 |
| region_Q_wait | 70 |
| region_dO_wait | 70 |
| region_dOt_wait | 96 |
| region_Qt_wait | 82 |
| region_S_retire | 29 |
| region_dP_retire | 23 |
| region_dV_retire_release | 27 |
| region_dK_retire_release_state | 221 |
| region_psi_barrier | 22 |
| vec_regions_wg0 | 1914 |
| vec_regions_wg0_in_wg1_outstanding | 0 |
| vec_regions_wg1 | 2174 |
| vec_regions_wg1_in_wg0_outstanding | 664 |
| p_region_dO_acquire | 127 |
| p_region_dOt_acquire_bracket | 2860 |
| p_region_Q_acquire_bracket | 186 |
| p_region_Qt_acquire_bracket | 87 |
| q_issue_lead | 4076 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 4%; WG1 in WG0's: 27%.

### D256 B4 H8/8 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3846 |
| step_wg1 | 3844 |
| offset | -44 |
| region_Q_wait | 78 |
| region_dO_wait | 68 |
| region_dOt_wait | 66 |
| region_Qt_wait | 82 |
| region_S_retire | 29 |
| region_dP_retire | 22 |
| region_dV_retire_release | 27 |
| region_dK_retire_release_state | 223 |
| region_psi_barrier | 22 |
| vec_regions_wg0 | 2001 |
| vec_regions_wg0_in_wg1_outstanding | 320 |
| vec_regions_wg1 | 1984 |
| vec_regions_wg1_in_wg0_outstanding | 614 |
| p_region_dO_acquire | 120 |
| p_region_dOt_acquire_bracket | 2889 |
| p_region_Q_acquire_bracket | 151 |
| p_region_Qt_acquire_bracket | 92 |
| q_issue_lead | 4056 |

Share of WG0's instrumented vector regions inside WG1's software-outstanding wgmma windows (issue to observed retire / release): 14%; WG1 in WG0's: 27%.


## What the sampled, instrumented records show (and what they do not)

1. In the sampled steps the consumers' load-wait regions are short (Q / dO / dOt / Qt: 60-100 cycles each) while the
   producer's acquire regions are long (880-2900 cycles per step) and Q(m+1) is issued ~4000 cycles before it is
   requested. In these records the consumers, not the loads, set the pace. This is sample- and instrumentation-
   specific evidence, not proof that load latency never matters in the plain kernel (the d256 dO-depth result shows it
   can).
2. The bracket from the dV retire/release marker (event 16) to the dK retire/release/state marker (event 17) is
   ~220-255 cycles per step at both head dims (from the dK issue marker, event 15: 269 / 280 cycles at D128 causal /
   full, 247 / 250 at D256), while the S / dP / dV retire regions are 9-30 cycles. Nothing independent sits between the dK issue and
   the dK fold. This tail region includes the release / state work and probe overhead; it is not a pure tensor stall,
   and how much of it a schedule change can recover is a hypothesis to measure (H1b).
3. Instrumented vector regions take ~2400-2600 (hd128) / ~1900-2200 (d256) of ~3850 cycles per warpgroup step.
4. The two warpgroups start each step within ~45-60 cycles of each other; 22-40% (hd128) / 4-27% (d256) of one's vector
   regions fall inside the other's software-outstanding wgmma windows. Consistent with aligned phases; it does not
   measure productive tensor / vector overlap, and whether staggering would shorten the step is not shown.
5. Missing boundaries (next bounded diagnostic build): an immediate S-issued marker (event 21 is the dO-wait entry,
   after S issue and scale prefetch), producer acquire entry markers for dOt / Q / Qt, consumer release boundaries
   separate from waits (d256 Q release sits inside the exp region), per-pipeline stage / phase (only Q's is recorded).

## Post-DO2 ranking (results/r3/ab_main_confirm/B_median.json, main grid, modeled SOL as recorded)

| kernel | measured ms | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|
| pass 1 | 492.003 | 128.810 | 363.193 | 26% |
| forward | 185.349 | 64.167 | 121.182 | 35% |
| pass 2 | 105.795 | 63.061 | 42.734 | 60% |
| torch glue | 36.445 | 13.822 | 22.623 | 38% |
| pass 0 | 35.649 | 16.042 | 19.607 | 45% |

## Timeline-supported ordering of the pass-1 hypotheses (historical, round 4)

Outcome since: H1b rejected in round 5 (-1.64% D128 pass 1); per Hao (round 5) the next pass-1 rounds are PINGPONG
(per-warpgroup psi + ping-pong) then MUFU-poly exp2; vector micro-reductions deprioritized. Current order and counters:
results/rounds.md.


1. **H1b (task5, bitwise, hd128)**: pre-pack the dS' transpose words before issuing the dK gemm, and run the shuffles,
   stmatrix, WG barriers and TMA store (~450 cycles per step of vector / sync regions in the dense profile) while dK is
   outstanding, aiming at the ~250-cycle dK tail region (benefit a hypothesis until measured). Requires the task4 Codex register-liveness / wgmma review first.
2. **Vector-work reductions (bitwise)**: psi amax via redux.sync on the |x| bit patterns; fewer instructions in the
   conversion / transpose paths. Each is small; candidates if H1b leaves the dK wait exposed.
3. **PINGPONG (non-bitwise; Q2 + Q3 first)**: stagger the warpgroups by half a step so one's vector work runs under the
   other's wgmma. The timeline confirms aligned phases; the benefit depends on issue contention, uncertain.
4. Not pursued: deeper hd128 load pipelines (the producer is not the limiter).
