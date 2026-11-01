# Pass 1 profile (hd128 P1-R and d256 V10), round 2

## How it was measured, and why not Nsight Compute

Nsight Compute, nsys GPU metrics and CUPTI PC sampling all need GPU performance counters, which the compute nodes
restrict to administrators (`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`; `ncu` fails with
ERR_NVGPUCTRPERM). Changing that needs root on the node (driver module parameter). So hardware stall reasons, pipe
utilization counters and L2 / smem throughput counters are not available here; this profile uses:

1. A profiling-only build (`DM_PROF=1` -> `-DKDA_PROF=1`, `csrc/kda_prof.h`; never shipped) whose pass 1 reads
   `clock64()` at 18 phase boundaries of each consumer warpgroup's m-block step and per tile / CTA, accumulates per
   warpgroup in 1 KB of static shared memory and flushes one `atomicAdd` per slot per CTA into a device buffer
   (`prof_buffer` op). With `KDA_PROF=0` the default build's SASS is identical to the baseline's (md5 of the
   address-stripped `cuobjdump -sass` of the whole library matches, job 21158748). The probe build's outputs are bitwise the
   baseline's (72 equivalence cases); a first version with a 512 B counter array was NOT (it moved the dynamic smem base
   off the 1024 B alignment the SW128 TMA layouts need: dK / dV relative error 1.4), which is why the array is 1 KB.
2. Perturbation: the probes cost about +20% pass-1 time (table below), so phase SHARES are the signal; absolute
   cycles are inflated by roughly that factor, spread over the probe points.
3. Resource usage (`cuobjdump --dump-resource-usage`), launch geometry and smem (`DM_DEBUG=1`), analytic budgets.
4. What-if builds (pipeline depth, warpgroup decoupling), timed against the baseline in the same job, 2 alternating
   repetitions (`results/r2/whatif_*.json`, patches in `results/r2/whatif_apply.py`).

## Resources and roles

- hd128 pass 1 (P1-R): 384 threads = 1 producer warpgroup (1 TMA warp, 24 regs) + 2 MMA warpgroups (240 regs via
  setmaxnreg; the static allocation reports REG 168 = 65536 / 384), 158.7 KB dynamic smem -> 1 CTA / SM, non-persistent
  LPT tile scheduler (grid = n-blocks x heads x batch), tile 64 query rows x 128 keys x 128 d, each MMA warpgroup owns
  64 keys. Stages: Q / Qt 2, dO / dOt 1 (certified QST 2, DOST 1, DOTST 1).
- d256 pass 1 (V10): 2 MMA warpgroups + producer + transposer roles, 213 KB smem -> 1 CTA / SM, tile 64 rows x 64 keys
  x 256 d; P~^T and dS'^T go through shared memory (exchange stores + 2 PdS barriers per step); Q stages 2, dO 1.
- STACK: 0 (causal) / 8 B (non-causal hd128) in the baseline; no LOCAL spills.

## Analytic budget per m-block step (one CTA)

- Tensor core: 4 wgmma of 2^21 FLOP per CTA step (S, dP, dV, dK) at both head dims = 8.4 MFLOP / 8192 FLOP / clk
  = **1024 cycles**.
- MUFU: one exp2 per score: 64 x 128 = 8192 per step (hd128) at 16 / clk / SM = **512 cycles** (d256: 4096 -> 256).
- HBM / L2: per hd128 step a CTA reads Q, Qt, dO, dOt tiles (4 x 8 KB, mostly L2 hits: every tile is re-read by the
  N / 128 CTAs of the head) and writes 8 KB of dS'. At the measured ~3300 plain cycles per step: about 0.6 TB/s of dS'
  writes and ~3 TB/s of L2 reads across 132 SMs, far from either limit: pass 1 is not memory-bound.

## Probe perturbation (pass 1 ms, plain baseline vs probe build)

| shape | plain | probes | change |
|---|---:|---:|---:|
| D128 B4 H16/16 N8192 causal | 3.882 | 4.667 | +20.2% |
| D128 B4 H16/16 N8192 full | 7.760 | 9.371 | +20.8% |
| D128 B1 H16/16 N32768 causal | 15.552 | 18.383 | +18.2% |
| D128 B4 H16/4 N8192 causal | 3.906 | 4.690 | +20.1% |
| D256 B4 H8/8 N8192 causal | 4.215 | 5.129 | +21.7% |
| D256 B4 H8/8 N8192 full | 8.177 | 10.143 | +24.0% |
| D256 B1 H8/8 N32768 causal | 16.839 | 20.027 | +18.9% |

## Per-phase cycles per m-block step (probe build)

Shares of the per-step loop; "wait X" phases are a warpgroup blocked on a pipeline barrier or on wgmma retirement.

### D128 B4 H16/16 N8192 causal (matched): pass 1 4.667 ms per call

m-block steps per call (WG0): 266240; cycles per step: WG0 3968, WG1 3967; tensor-core ideal 1024 -> tensor busy <= 26%; consumer loops cover 88% of the kernel's SM-cycles (ideal tensor share 23%)

per tile (CTA): lifetime 264924 cycles, prologue 2028, exit 586, epilogue (dK/dV store) 782

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 208 | 5.2% | 220 |
| issue S + wait dO + issue dP | 399 | 10.1% | 289 |
| wait S (wgmma) | 165 | 4.2% | 128 |
| mask + exp2 + release Q | 647 | 16.3% | 744 |
| wait dP (wgmma) | 74 | 1.9% | 63 |
| dS line + release dO | 219 | 5.5% | 193 |
| psi amax + shuffles | 314 | 7.9% | 346 |
| psi barrier (cross-WG) | 38 | 1.0% | 39 |
| psi math + P / dS' fp8 conversion | 426 | 10.7% | 454 |
| dS' transpose shfl / prmt | 120 | 3.0% | 125 |
| store drain + WG barriers + stmatrix + TMA | 330 | 8.3% | 305 |
| dV rescale + wait dOt + issue dV | 224 | 5.6% | 233 |
| wait Qt + issue dK | 158 | 4.0% | 161 |
| wait dV (wgmma) + release | 58 | 1.5% | 47 |
| wait dK (wgmma) + release | 226 | 5.7% | 198 |
| dK fold + dV promote | 361 | 9.1% | 421 |

### D128 B4 H16/16 N8192 full (stale): pass 1 9.371 ms per call

m-block steps per call (WG0): 524288; cycles per step: WG0 4036, WG1 4035; tensor-core ideal 1024 -> tensor busy <= 25%; consumer loops cover 88% of the kernel's SM-cycles (ideal tensor share 22%)

per tile (CTA): lifetime 522378 cycles, prologue 1434, exit 589, epilogue (dK/dV store) 664

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 202 | 5.0% | 218 |
| issue S + wait dO + issue dP | 406 | 10.1% | 348 |
| wait S (wgmma) | 153 | 3.8% | 126 |
| mask + exp2 + release Q | 697 | 17.3% | 749 |
| wait dP (wgmma) | 70 | 1.7% | 61 |
| dS line + release dO | 212 | 5.3% | 191 |
| psi amax + shuffles | 314 | 7.8% | 340 |
| psi barrier (cross-WG) | 38 | 0.9% | 39 |
| psi math + P / dS' fp8 conversion | 435 | 10.8% | 434 |
| dS' transpose shfl / prmt | 123 | 3.0% | 127 |
| store drain + WG barriers + stmatrix + TMA | 334 | 8.3% | 322 |
| dV rescale + wait dOt + issue dV | 228 | 5.7% | 236 |
| wait Qt + issue dK | 154 | 3.8% | 159 |
| wait dV (wgmma) + release | 57 | 1.4% | 48 |
| wait dK (wgmma) + release | 239 | 5.9% | 206 |
| dK fold + dV promote | 373 | 9.2% | 432 |

### D128 B1 H16/16 N32768 causal (matched): pass 1 18.383 ms per call

m-block steps per call (WG0): 1052672; cycles per step: WG0 3954, WG1 3954; tensor-core ideal 1024 -> tensor busy <= 26%; consumer loops cover 88% of the kernel's SM-cycles (ideal tensor share 23%)

per tile (CTA): lifetime 1023277 cycles, prologue 1754, exit 585, epilogue (dK/dV store) 773

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 204 | 5.2% | 217 |
| issue S + wait dO + issue dP | 398 | 10.1% | 283 |
| wait S (wgmma) | 165 | 4.2% | 127 |
| mask + exp2 + release Q | 644 | 16.3% | 745 |
| wait dP (wgmma) | 74 | 1.9% | 64 |
| dS line + release dO | 218 | 5.5% | 193 |
| psi amax + shuffles | 314 | 7.9% | 346 |
| psi barrier (cross-WG) | 38 | 1.0% | 39 |
| psi math + P / dS' fp8 conversion | 426 | 10.8% | 457 |
| dS' transpose shfl / prmt | 119 | 3.0% | 124 |
| store drain + WG barriers + stmatrix + TMA | 331 | 8.4% | 304 |
| dV rescale + wait dOt + issue dV | 223 | 5.6% | 232 |
| wait Qt + issue dK | 158 | 4.0% | 162 |
| wait dV (wgmma) + release | 58 | 1.5% | 46 |
| wait dK (wgmma) + release | 225 | 5.7% | 197 |
| dK fold + dV promote | 357 | 9.0% | 416 |

### D128 B4 H16/4 N8192 causal (matched): pass 1 4.690 ms per call

m-block steps per call (WG0): 266240; cycles per step: WG0 3968, WG1 3968; tensor-core ideal 1024 -> tensor busy <= 26%; consumer loops cover 88% of the kernel's SM-cycles (ideal tensor share 23%)

per tile (CTA): lifetime 264917 cycles, prologue 1998, exit 586, epilogue (dK/dV store) 782

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 208 | 5.2% | 220 |
| issue S + wait dO + issue dP | 400 | 10.1% | 289 |
| wait S (wgmma) | 164 | 4.1% | 128 |
| mask + exp2 + release Q | 647 | 16.3% | 744 |
| wait dP (wgmma) | 74 | 1.9% | 63 |
| dS line + release dO | 219 | 5.5% | 193 |
| psi amax + shuffles | 314 | 7.9% | 346 |
| psi barrier (cross-WG) | 38 | 1.0% | 39 |
| psi math + P / dS' fp8 conversion | 426 | 10.7% | 454 |
| dS' transpose shfl / prmt | 120 | 3.0% | 125 |
| store drain + WG barriers + stmatrix + TMA | 330 | 8.3% | 306 |
| dV rescale + wait dOt + issue dV | 224 | 5.6% | 233 |
| wait Qt + issue dK | 158 | 4.0% | 161 |
| wait dV (wgmma) + release | 58 | 1.5% | 47 |
| wait dK (wgmma) + release | 227 | 5.7% | 198 |
| dK fold + dV promote | 361 | 9.1% | 422 |

### D256 B4 H8/8 N8192 causal (matched): pass 1 5.129 ms per call

m-block steps per call (WG0): 264192; cycles per step: WG0 4414, WG1 4414; tensor-core ideal 1024 -> tensor busy <= 23%; consumer loops cover 88% of the kernel's SM-cycles (ideal tensor share 20%)

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 235 | 5.3% | 235 |
| issue S + wait dO + issue dP | 803 | 18.2% | 797 |
| wait S (wgmma) | 93 | 2.1% | 103 |
| release Q + exp2 | 369 | 8.4% | 367 |
| wait dP (wgmma) + release dO | 56 | 1.3% | 54 |
| dS line | 85 | 1.9% | 86 |
| psi amax + slot publish | 279 | 6.3% | 278 |
| psi barrier (cross-WG) | 38 | 0.9% | 38 |
| psi math + P conversion + store drain + PdS barrier | 354 | 8.0% | 348 |
| P~ / dS' exchange stores (smem) | 453 | 10.3% | 437 |
| PdS barrier + TMA dS' store | 256 | 5.8% | 217 |
| dV rescale + wait dOt + issue dV | 290 | 6.6% | 318 |
| wait Qt + issue dK | 255 | 5.8% | 290 |
| wait dV (wgmma) + release | 102 | 2.3% | 99 |
| wait dK (wgmma) + releases | 153 | 3.5% | 156 |
| dK fold + dV promote | 591 | 13.4% | 591 |

### D256 B4 H8/8 N8192 full (stale): pass 1 10.143 ms per call

m-block steps per call (WG0): 524288; cycles per step: WG0 4391, WG1 4391; tensor-core ideal 1024 -> tensor busy <= 23%; consumer loops cover 88% of the kernel's SM-cycles (ideal tensor share 21%)

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 247 | 5.6% | 248 |
| issue S + wait dO + issue dP | 786 | 17.9% | 782 |
| wait S (wgmma) | 102 | 2.3% | 101 |
| release Q + exp2 | 358 | 8.1% | 343 |
| wait dP (wgmma) + release dO | 49 | 1.1% | 45 |
| dS line | 84 | 1.9% | 84 |
| psi amax + slot publish | 289 | 6.6% | 289 |
| psi barrier (cross-WG) | 37 | 0.8% | 38 |
| psi math + P conversion + store drain + PdS barrier | 393 | 9.0% | 414 |
| P~ / dS' exchange stores (smem) | 425 | 9.7% | 379 |
| PdS barrier + TMA dS' store | 222 | 5.0% | 191 |
| dV rescale + wait dOt + issue dV | 287 | 6.5% | 343 |
| wait Qt + issue dK | 254 | 5.8% | 307 |
| wait dV (wgmma) + release | 108 | 2.5% | 92 |
| wait dK (wgmma) + releases | 166 | 3.8% | 165 |
| dK fold + dV promote | 584 | 13.3% | 570 |

### D256 B1 H8/8 N32768 causal (matched): pass 1 20.027 ms per call

m-block steps per call (WG0): 1050624; cycles per step: WG0 4400, WG1 4400; tensor-core ideal 1024 -> tensor busy <= 23%; consumer loops cover 90% of the kernel's SM-cycles (ideal tensor share 21%)

| phase | WG0 cycles / step | share | WG1 cycles / step |
|---|---:|---:|---:|
| wait Q stage | 226 | 5.1% | 226 |
| issue S + wait dO + issue dP | 804 | 18.3% | 796 |
| wait S (wgmma) | 93 | 2.1% | 103 |
| release Q + exp2 | 369 | 8.4% | 367 |
| wait dP (wgmma) + release dO | 56 | 1.3% | 54 |
| dS line | 85 | 1.9% | 86 |
| psi amax + slot publish | 279 | 6.3% | 279 |
| psi barrier (cross-WG) | 38 | 0.9% | 38 |
| psi math + P conversion + store drain + PdS barrier | 355 | 8.1% | 349 |
| P~ / dS' exchange stores (smem) | 454 | 10.3% | 438 |
| PdS barrier + TMA dS' store | 260 | 5.9% | 222 |
| dV rescale + wait dOt + issue dV | 290 | 6.6% | 318 |
| wait Qt + issue dK | 255 | 5.8% | 291 |
| wait dV (wgmma) + release | 102 | 2.3% | 99 |
| wait dK (wgmma) + releases | 153 | 3.5% | 156 |
| dK fold + dV promote | 580 | 13.2% | 579 |

## What-if measurements (pass 1 ms change vs baseline, mean of 2 alternating reps, same job)

| variant | D128 N8k causal | D128 N8k full | D128 N32k causal | D128 N8k GQA4 | D256 N8k causal | D256 N8k full | D256 N32k causal | bitwise |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| d256 dO stages 1 -> 2 (smem 213 -> 224 KB) | +0.1% | -0.0% | -0.3% | +0.1% | **-5.0%** | **-2.4%** | **-3.5%** | yes (72 cases) |
| hd128 dO / dOt stages 1 -> 2 (certified DOST / DOTST) | +1.7% | -0.0% | +1.7% | +1.7% | +0.1% | +0.0% | +0.2% | yes |
| hd128 Q / Qt stages 2 -> 3 (certified QST) | +1.0% | +0.4% | +1.3% | +1.1% | -0.0% | +0.0% | +0.4% | yes |
| hd128 warpgroup-local psi, no cross-WG barrier (TIMING PROBE, numerics wrong) | **-4.8%** | -2.7% | **-4.2%** | **-4.8%** | +0.0% | +0.5% | +1.0% | no |

## Reading

1. **Pass 1 is issue-bound and serialized, not memory-bound.** A step takes ~3970 probe cycles (~3300 plain) against a
   1024-cycle tensor ideal: the tensor pipe is busy at most 23-26% of the time. No single phase dominates (largest:
   exp2 16% at hd128, issue S/dP + wait dO 18% at d256, dK fold + dV promote 9-13%, conversions 11%, amax 8%, store
   8%). Everything inside a warpgroup is a dependent chain: wgmma issue -> wait -> elementwise -> conversions ->
   wgmma, and the two warpgroups run in lockstep (the per-tile psi barrier costs only ~38 cycles itself, but it keeps the
   two warpgroups' elementwise phases aligned, so tensor work of one never overlaps vector work of the other).
2. **The MUFU floor is half the tensor ideal at hd128** (512 of 1024 cycles per step): even with perfect overlap,
   exp2 alone would occupy the SFU for half the step; FA3's fp8 kernels face the same limit.
3. **Load waits:** "wait Q" ~5% at both head dims; at d256 "issue S + wait dO + issue dP" is 18% and a second dO stage
   removes 2.4-5% of pass 1 (bitwise): the single dO stage exposes TMA latency there. At hd128 deeper stages are slower
   (one CTA per SM already; more smem stages lengthen the per-tile prologue).
4. **Decoupling the two warpgroups alone (per-warpgroup psi) gives ~4-5% at hd128**; the remaining overlap needs an
   explicit ping-pong schedule (FA3-style named-barrier ordering so one warpgroup's exp / conversion runs under the
   other's wgmma), and per-warpgroup psi is a numerics change (two scales per 128-key tile; pass 2 must consume two
   psi per tile).

## Profile-based hypotheses for pass 1 (ranked by expected gain x confidence)

1. **P1-d256-DO2** (d256, bitwise, schedule): double-buffer dO in the d256 pass 1 (`Stages_dO_d256 = 2`, smem 224 KB
   of 227 KB). Measured -2.4 .. -5.0% of d256 pass 1 in the what-if; next round, as an acceptance candidate on the
   D256 subset.
2. **P1-hd128-PINGPONG** (hd128, structural, non-bitwise): per-warpgroup psi (the stripped M23 HALFPSI arm: psi per
   64 keys, finer scale granularity) re-implemented in pass 1 and pass 2, plus explicit ping-pong ordering of the two
   warpgroups' elementwise phases against each other's wgmma. Upper bound from the probe without ping-pong: -4..-5%;
   ping-pong targets the lockstep that leaves the tensor pipe idle during exp / conversions (up to ~2x headroom in
   principle). Needs the AC-2.3 numerics tooling (Q2).
3. **P1-MUFU** (both, non-bitwise): compute part of the exp2 with an FMA polynomial (FA4-style) to offload the SFU; P is
   quantized to e4m3 afterwards, so few codes move; needs the numerics tooling.
4. **P1-AMAX-REDUX** (both, bitwise, small): psi amax via `redux.sync.max.u32` on the |x| bit patterns (exact for
   non-negative floats) instead of 5 shuffle+max steps; ~2-3% of the step.
5. **P1-FOLD** (both): the per-tile dK fold (64 FFMA / thread / step) plus the dV promote are 9-13%; options: fold two
   tiles per pass when the g basis is equal (bitwise when the ratio is exactly 1), or fold into smem asynchronously.
6. **Structural, pass 1 + pass 2**: the dS' round trip (pass 1 writes 8 KB per step, pass 2 re-reads it and is
   memory-bound at D 128): a deterministic fused dQ (per-row-block ownership with ordered accumulation) would remove
   pass 2's HBM stream; a large redesign of the pass-2 numerics (windowed promotion), deferred behind 1-5.
