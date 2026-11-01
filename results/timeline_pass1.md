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

Consumer waits are pipeline waits measured from wait entry to exit; wgmma waits are warpgroup_wait durations; producer
"p_acq" are producer-acquire waits (the producer blocked on a stage the consumers have not released); q_ready_lead is
how long before the consumers asked for Q(m+1) the producer had issued it. "vec" is a warpgroup's vector work (exp,
dS, amax, conversions, transpose / exchange, store, fold), "under other mma" the part of it during which the other
warpgroup had a wgmma in flight (issue to retire).

### D128 B4 H16/16 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3849 |
| step_wg1 | 3866 |
| offset | -61 |
| wait_Q | 82 |
| wait_dO | 80 |
| wait_dOt | 62 |
| wait_Qt | 85 |
| wait_S | 9 |
| wait_dP | 9 |
| wait_dV | 20 |
| wait_dK | 253 |
| psi_barrier | 7 |
| vec_wg0 | 2471 |
| vec_wg0_under_wg1_mma | 708 |
| vec_wg1 | 2574 |
| vec_wg1_under_wg0_mma | 1078 |
| p_acq_dO | 878 |
| p_acq_dOt | 1613 |
| p_acq_Q | 148 |
| p_acq_Qt | 142 |
| q_ready_lead | 3978 |

Share of WG0's vector work while WG1 has wgmma in flight: 27%; WG1 under WG0's wgmma: 40%.

### D128 B4 H16/16 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3938 |
| step_wg1 | 3944 |
| offset | -49 |
| wait_Q | 65 |
| wait_dO | 84 |
| wait_dOt | 64 |
| wait_Qt | 82 |
| wait_S | 26 |
| wait_dP | 9 |
| wait_dV | 23 |
| wait_dK | 255 |
| psi_barrier | 7 |
| vec_wg0 | 2398 |
| vec_wg0_under_wg1_mma | 534 |
| vec_wg1 | 2608 |
| vec_wg1_under_wg0_mma | 1022 |
| p_acq_dO | 986 |
| p_acq_dOt | 1674 |
| p_acq_Q | 152 |
| p_acq_Qt | 144 |
| q_ready_lead | 4056 |

Share of WG0's vector work while WG1 has wgmma in flight: 22%; WG1 under WG0's wgmma: 37%.

### D256 B4 H8/8 N8192 causal (matched): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3801 |
| step_wg1 | 3801 |
| offset | -61 |
| wait_Q | 70 |
| wait_dO | 70 |
| wait_dOt | 96 |
| wait_Qt | 82 |
| wait_S | 29 |
| wait_dP | 23 |
| wait_dV | 27 |
| wait_dK | 221 |
| psi_barrier | 22 |
| vec_wg0 | 1914 |
| vec_wg0_under_wg1_mma | 0 |
| vec_wg1 | 2174 |
| vec_wg1_under_wg0_mma | 664 |
| p_acq_dO | 127 |
| p_acq_dOt | 2860 |
| p_acq_Q | 186 |
| p_acq_Qt | 87 |
| q_ready_lead | 4076 |

Share of WG0's vector work while WG1 has wgmma in flight: 4%; WG1 under WG0's wgmma: 27%.

### D256 B4 H8/8 N8192 full (stale): 8 sampled CTAs

80 interior steps; medians in cycles (SM clock):

| quantity | median |
|---|---:|
| step_wg0 | 3846 |
| step_wg1 | 3844 |
| offset | -44 |
| wait_Q | 78 |
| wait_dO | 68 |
| wait_dOt | 66 |
| wait_Qt | 82 |
| wait_S | 29 |
| wait_dP | 22 |
| wait_dV | 27 |
| wait_dK | 223 |
| psi_barrier | 22 |
| vec_wg0 | 2001 |
| vec_wg0_under_wg1_mma | 320 |
| vec_wg1 | 1984 |
| vec_wg1_under_wg0_mma | 614 |
| p_acq_dO | 120 |
| p_acq_dOt | 2889 |
| p_acq_Q | 151 |
| p_acq_Qt | 92 |
| q_ready_lead | 4056 |

Share of WG0's vector work while WG1 has wgmma in flight: 14%; WG1 under WG0's wgmma: 27%.



## What the timeline shows (instrumented kernel)

1. Loads are not the limiter in the sampled steps: consumer waits for Q / dO / dOt / Qt are 60-100 cycles each, while the
   PRODUCER waits 880-2900 cycles per step to acquire a stage (it runs ahead and is throttled by consumer releases);
   Q(m+1) is issued ~4000 cycles before it is needed. Consistent with the round-2 what-if (deeper hd128 stages slower).
2. wgmma retirement waits are short except one: S / dP / dV waits 9-30 cycles, but the dK wait right before the dK fold
   is ~220-255 cycles per step (~6% of the step) at both head dims: nothing independent is scheduled between the dK
   issue and its fold.
3. Vector work dominates each warpgroup's step: ~2400-2600 of ~3850 cycles at hd128, ~1900-2200 at d256.
4. The two warpgroups start each step within ~45-60 cycles of each other (aligned); only 22-40% (hd128) and 4-27% (d256)
   of one warpgroup's vector work runs while the other has a wgmma in flight. The psi barrier itself costs 7-22 cycles.
   This supports the lockstep hypothesis (aligned phases); whether staggering would shorten the step depends on whether
   the two warpgroups' vector work is issue-limited when co-scheduled, which this trace cannot show.

## Post-DO2 ranking (results/r3/ab_main_confirm/B_median.json, main grid, modeled SOL as recorded)

| kernel | measured ms | modeled SOL ms | modeled waste ms | % of modeled SOL |
|---|---:|---:|---:|---:|
| pass 1 | 492.003 | 128.810 | 363.193 | 26% |
| forward | 185.349 | 64.167 | 121.182 | 35% |
| pass 2 | 105.795 | 63.061 | 42.734 | 60% |
| torch glue | 36.445 | 13.822 | 22.623 | 38% |
| pass 0 | 35.649 | 16.042 | 19.607 | 45% |

## Timeline-supported ordering of the remaining pass-1 hypotheses

1. **H1b (task5, bitwise, hd128)**: pre-pack the dS' transpose words before issuing the dK gemm, and run the shuffles,
   stmatrix, WG barriers and TMA store (~450 cycles per step of vector / sync work in the dense profile) while dK is in
   flight, filling the exposed ~250-cycle dK wait. Requires the task4 Codex register-liveness / wgmma review first.
2. **Vector-work reductions (bitwise)**: psi amax via redux.sync on the |x| bit patterns; fewer instructions in the
   conversion / transpose paths. Each is small; candidates if H1b leaves the dK wait exposed.
3. **PINGPONG (non-bitwise; Q2 + Q3 first)**: stagger the warpgroups by half a step so one's vector work runs under the
   other's wgmma. The timeline confirms aligned phases; the benefit depends on issue contention, uncertain.
4. Not pursued: deeper hd128 load pipelines (the producer is not the limiter).
