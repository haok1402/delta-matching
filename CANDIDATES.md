# kda-solcheck: are the sol4 floors reachable? (branch kda-sm90-solcheck)

Helper kda-solcheck, branch `kda-sm90-solcheck` on origin/kda-sm90 (rebased on 70b482e). This helper ships no kernel change. Its
deliverable is a measurement: synthetic microkernels with the exact step-loop instruction mix of the accepted kernels
(frozen r17_p0s), the floors' assumptions measured directly, and a corrected floor. The full report is
`results/solcheck/README.md`; the tools are in `solcheck/`.

## Conclusion for kda-sm90

90% of sol4-v2 is NOT reachable for the forward, pass 1 or pass 0 with their current step-loop instruction mixes on
this H100. An ideal kernel with exactly that mix reaches at most **70% / 62% / 75%** of sol4. That is the best of five
schedules (real order, no dependence, interleaved, warp-specialized, ping-pong), with the same warpgroups and
registers, random operands and no producer, TMA, masking, epilogue or scheduler costs.

Two fresh jobs, real kernels and ideal in the same job and calibration (21183777, 21183778):

| kernel | real % of sol4 | ideal-mix ceiling % of sol4 | real / ideal (calibration-independent) |
|---|---:|---:|---:|
| forward | 57.6 / 57.5 | 69.7 / 70.1 | 0.827 / 0.820 |
| pass 1 | 42.5 / 42.3 | 62.0 / 62.1 | 0.685 / 0.681 |
| pass 0 | 69.8 / 69.7 | 74.5 / 74.6 | 0.937 / 0.935 |

Why the ceiling sits there (measured):
- **wgmma runs at the full 8192 FLOP/clk/SM** in every shape used (SS n64 / n128 / n256, RS n128). Under the 700 W
  power cap that is 1.62-1.70 PFLOP/s, 17-24% above the cuBLASLt calibration. The tensor floor is lenient.
- **The ALU mixes alone are already near their bound** without tensor work: forward and pass 0 MUFU-bound at 91-94%,
  pass 1 issue-bound at 88-92%. IPC is 3.52-3.57 per SM, so 4 per SM per clock is not reachable. ALU-pipe ops issue at
  0.5 per SMSP per clock, SHFL at 0.25, and there is no dual issue.
- **The floors are coupled:**
  - While wgmma streams at full rate, the other warps on the SM lose about half of their issue capacity: FFMA / LDS /
    SHFL 2.0x slower, FMNMX / IMAD 1.65x, MUFU >= 1.36x.
  - This holds with fewer ALU warps or twice the wgmma instructions, and was reproduced cycle for cycle in two jobs.
  - Result: the time per step is about C_alu + 0.55 x C_tensor (cycles), not max(tensor, MUFU, issue).
- **The effective SM clock under tensor load is 1.66-1.80 GHz** (clock64 / globaltimer). nvidia-smi reports
  1815-1980 MHz and misses the dips. sol4's MUFU / issue floors use the nvidia-smi clock, so they are 10-17% optimistic.
- **wgmma costs:** a drain-to-zero wait costs about 60-70 cycles; a ping-pong barrier handoff about 37-39 cycles.
- **Sensitivity (real dependence order):**
  - pass 0 gains about 0.16 percentage points of sol4 per instruction removed per warp-step (still < 85% at -150);
  - pass 1 about 0.05 points;
  - forward gains 1.6 points with no MUFU at all; its order and the coupling bind, not MUFU.

## Proposed corrected floor (for the gate): sol4c = sol4-v2 floor / ideal fraction

- Ideal fractions: forward 0.699, pass 1 0.620, pass 0 0.745 (`results/solcheck/factors.json`, the r17_p0s mix);
  pass 2 unchanged (bytes floor, 94-95%).
- Tool: `python solcheck/sol4c.py --sol4 <sol4v2.json>`. Accepted r17_p0s: forward 83.3%, pass 1 68.9%, pass 0 94.8%
  of sol4c.
- **Pass 0 meets the 90% goal under sol4c.**
- 90% of sol4c means about 63% of sol4 for the forward (about 61 ms on the four shapes vs 66 now) and about 56% for
  pass 1 (about 137 ms vs 179 now).
- Measure real and ideal in the same job (`solcheck/run_job.sh`, REAL=1). A candidate that changes its loop's mix must
  re-derive its own ideal (`solcheck/mix.py` -> `cfgs.py` -> `build_job.sh` -> `run_job.sh`).
- Repairs to sol4 itself (details in the report, section 4):
  - MUFU / issue floors at the effective clock;
  - per-pipe issue rates;
  - the coupling term (kappa ≈ 0.55);
  - same-job ratios or a multi-job median for the tensor-peak calibration, which moves +-5% between jobs.
  - With these, the model predicts the ideal kernels within 1-5%.

## Entries

### SC1: synthetic ideal-mix step kernels + floor-assumption microbenchmarks (measurement, no kernel change)

- Commits on `kda-sm90-solcheck`: f1b7dac (generator, microbenchmarks, run r3) and the commit adding this file.
- What it changes: nothing in the library (`solcheck/`, `results/solcheck/`, `CANDIDATES.md` only); bitwise n/a.
- Numbers: the tables above. Jobs:
  - 21183317 (r3);
  - 21183628 (r4, interference);
  - 21183777 / 21183778 (r5 / r6, real + ideal, same job);
  - 21183994 (r7, interference repeat);
  - 21184066 (r8, sweeps).
- Resources of the synthetic kernels:
  - forward 159-161 registers, 3 consumer warpgroups (real: 160 via setmaxnreg);
  - pass 1 234-246 / 2 warpgroups (real 240);
  - pass 0 90-96 / 4 warpgroups (real 104);
  - STACK 0, no spills, no ptxas wgmma serialization (C75xx) in any kernel used for a conclusion. The three
    double-buffered 3-4 warpgroup tensor probes that do serialize are excluded.
- Status: n/a (report for kda-sm90's gate and floor decision; no READY / REJECTED kernel candidate).

### SC2: proposed corrected floor sol4c (`solcheck/sol4c.py`, `results/solcheck/factors.json`)

- What it changes: a gate definition, not code in the library. Up to kda-sm90 / Hao.
- Status: PROPOSAL.

## Final summary

- All levers in scope were measured with at least two designed variants each:
  - concurrency: dep, free, ilv, spec, pp;
  - interference: 6 pipe classes, 2 warp counts, 2 wgmma shapes;
  - clock and power: random vs zero operands, sustained runs;
  - waits and barriers: 1 / 4 wgmma, 1 / 2 warpgroups, ping-pong with 2 / 3 warpgroups;
  - issue: single pipes, pipe pairs, mixes alone;
  - sweeps: instruction count, MUFU share.
- No synthetic schedule of these mixes exceeds 70% (forward), 62% (pass 1) or 75% (pass 0) of sol4.
- What is left for the kernel helpers, measured against the ideal of each mix:
  - pass 1: 32% (the largest gap, not explained by its loop's mix);
  - forward: 18%;
  - pass 0: 6%.
