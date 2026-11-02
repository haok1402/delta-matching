# Are the sol4 floors reachable? Ideal-mix synthetic step kernels (kda-solcheck)

Question (Hao / kda-sm90): the goal is forward, pass 0, pass 1 and pass 2 each at >= 90% of the calibrated sol4-v2
limit; the helpers stall at pass 1 ~43%, forward ~58%, pass 0 ~70%. Is 90% reachable at all with the instruction
mix these kernels execute, on this H100?

Answer: **no, for all three compute kernels.** An ideal kernel with exactly the step-loop instruction mix of the accepted
kernels reaches at most **70% (forward), 62% (pass 1), 75% (pass 0)** of sol4. That holds on this H100 in the best of
five schedules: the real dependence order, no dependence, fine interleave, warp specialization and ping-pong. It holds
with the same warpgroups and register footprint, with no producer, TMA, masking, epilogue or scheduler costs, and with
random fp8 operands. Measured against that ideal, the real kernels stand at **82% (forward), 68% (pass 1), 94%
(pass 0)**. Pass 0 is already past 90% of what its mix allows.

Pass 2 (bytes-bound, 94-95% of sol4) was not in question and is not covered here.

## Method

- `solcheck/mix.py`: the opcode histogram of each kernel's sol4-v2 step loop along the path sol4 charges (cheapest
  full-work path), from the frozen r17_p0s library (`solcheck/mix_r17p0s.json`). Per warp per step:
  - **Forward (BM192, 3 consumer WGs, 160 regs):** 599 instructions. 4 QGMMA 64x128x32 SS (S = QK^T) and 4 RS (PV);
    74 EX2, 1 RCP, 140 FFMA, 85 FMUL, 72 FADD, 82 FMNMX, 32 F2FP, 8 SHFL; the rest integer, uniform and control.
  - **Pass 1 (2 consumer WGs, 240 regs):** 658 instructions. 8 QGMMA 64x64x32 SS (S, dP) and 4 RS 64x128x32
    (dV, dK); 32 EX2 + 3 RCP, 128 FFMA, 107 FMUL, 82 FMNMX, 32 FADD, 32 F2FP, 21 SHFL, 18 LDS.64.
  - **Pass 0 (W4, 4 consumer WGs, 104 regs):** 269 instructions. 8 QGMMA 64x64x32 SS; 32 EX2, 64 FFMA, 37 FMUL,
    36 FADD, 32 FMNMX.
- `solcheck/gen.py` + `cfgs.py`: a generator of persistent synthetic kernels (one CTA per SM, inline PTX wgmma) whose
  step loop issues the same wgmma shapes, counts and SS / RS operand sources, and the same MUFU / FFMA / FMUL / FADD /
  FMNMX / F2FP / SHFL / LDS counts.
  - Integer, uniform and control ops are emitted as LOP3 / IMAD filler. The filler count is adjusted until the
    compiled loop has exactly the real instruction count (599 / 658 / 269, `solcheck/check.py`).
  - Same warpgroup count and register footprint (forward 159-161 registers, as the real 160). No ptxas wgmma
    serialization: every synthetic used here compiles without C75xx.
  - Random e4m3 operands in shared memory. All-zero operands draw 255-400 W instead of the 690 W the real kernels draw
    (tensor-core power depends on operand toggling).
  - Zero-init decided at run time, so ptxas cannot remove ALU work as dead.
  - Structures:
    - dep: the real step order with its waits. Forward: PV(RS) -> wait -> O += PV (64 FADD) -> S(SS) -> wait ->
      softmax -> P. Pass 1: S, dP -> wait -> P -> wait -> dS -> dV, dK -> wait. Pass 0: S, dP -> wait -> ALU.
    - free: no data dependence; ALU work on private registers.
    - ilv: wgmma interleaved through the ALU work.
    - spec: warpgroup 0 issues all tensor work, the other warpgroups run the ALU mix.
    - pp: forward with FA3 named-barrier ping-pong.
- `solcheck/micro.py`: assumption microbenchmarks.
  - wgmma rate per shape and warpgroup count.
  - Per-pipe rates.
  - The ALU mixes alone.
  - wgmma + MUFU / FFMA / FMNMX.
  - Interference: one warpgroup streams wgmma, three others run one pipe class.
  - wgmma wait round trips and ping-pong barriers.
- `solcheck/sweep.py`: sensitivity of the dep structure to instructions and MUFU per step.
- Timing: `solcheck/run_job.sh` (1 GPU; `calibrate.py` in the same job).
  - CUDA events over 20 launches x 5 reps, about 2 ms per launch.
  - clock64 / globaltimer stamps per warpgroup, giving cycles and the effective SM clock.
  - nvidia-smi sampled every 50 ms, giving power, clock and throttle reasons.
  - The real kernels (`kernels.py --grid paper`, frozen r17_p0s, `sol4.py model`) in the same job, so real and ideal
    use the same calibration.
- Analysis: `solcheck/analyze.py`, `summary.py`, `decompose.py`, `smi_phases.py`. Raw data is in `r3`, `r5`-`r8`
  (job ids in each `calib.json`).

## 1. Real vs ideal, same job (r5 = 21183777, r6 = 21183778; calibrated GEMM 1437 / 1435 TFLOP/s)

| kernel | real r17_p0s % of sol4 | dep | free | ilv | spec | pp | best ideal | real / ideal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forward | 57.6 / 57.5 | 64.9 / 64.5 | 69.2 / 68.8 | 65.8 / 65.9 | **69.7 / 70.1** | 66.8 / 66.7 | 69.9% | **0.823** |
| pass 1 | 42.5 / 42.3 | 61.7 / 60.9 | 58.7 / 58.6 | **62.0 / 62.1** | 61.4 / 61.7 | - | 62.0% | **0.683** |
| pass 0 | 69.8 / 69.7 | 71.2 / 70.8 | 74.0 / 74.6 | **74.5 / 74.3** | 74.2 / 74.5 | - | 74.5% | **0.936** |

The same ranking shows in r3 (21183317, calibration 1381: forward 72.7, pass 1 64.7, pass 0 78.6%) and in r4
(21183628, calibration 1513: 66.3 / 59.6 / 71.7%). The calibrated GEMM peak moves +-5% between jobs and nodes, and
every % of sol4 moves with it. Same-job ratios (real / ideal) do not.

## 2. Why the ideal stalls at 62-75%

All numbers are per step per SM, in SM cycles, mean of r5 and r6 (`decompose_r5r6.md`). C_tensor is the tensor cycles
at the full 8192 FLOP/clk; C_alu is the ALU mix alone on the same warpgroups (two estimates: the ALU-only build of the
mix and the warp-specialized build with its tensor work removed).

| kernel | C_tensor | C_alu alone | its bound | C_best | kappa | f_eff (clock64/ns) | 8192 f_eff / calibrated peak | C_tensor / C_best |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| forward | 1536 | 1974-2085 | MUFU (94%) | 2909 | 0.54 | 1789 MHz | 1.35 | 0.53 |
| pass 1 | 1024 | 1377-1529 | issue (92%) | 2163 | 0.62 | 1780 MHz | 1.34 | 0.47 |
| pass 0 | 1024 | 1131-1192 | MUFU (94%) | 1724 | 0.52 | 1691 MHz | 1.27 | 0.59 |

Fraction of sol4 = (8192 f_eff / calibrated peak per SM) x (C_tensor / C_best).

The first factor is about 1.3: the tensor floor is lenient. wgmma runs at the full 8192 FLOP/clk, and at the
effective clock that is 27-35% above the cuBLASLt calibration.

The loss is in the second factor. C_best ≈ C_alu + kappa x C_tensor with kappa = 0.52-0.62: every tensor cycle costs
half a cycle or more of ALU-mix throughput. The three floors are not independent, as max(tensor, MUFU, issue) assumes.

### 2a. Tensor pipe (t_*, r3-r6)

- **Every shape reaches the full dense rate.** wgmma SS n64 / n128 / n256 and RS n128 all run at 8191-8192 FLOP/clk/SM:
  - from 2 warpgroups on, with one wait per 8-wgmma group;
  - and with 1 warpgroup that keeps a group in flight (two accumulators, wait_group 1).
- **SS n64 is not limited by shared-memory bandwidth:** 4 warpgroups reach 8191/clk too.
- **Each kernel's own tensor groups**, with their waits and no ALU work, reach 8079-8192 FLOP/clk.
- **Wall-clock rate with random operands:**
  - 1.62-1.70 PFLOP/s at the 700 W cap: power ~687 W, nvidia-smi 1665-1725 MHz, clock64/ns 1515-1650 MHz.
  - That is 117-124% of the cuBLASLt calibration.
  - With zero operands: 255 W and 1.70-1.82 PFLOP/s.
- **Latency:**
  - One SS wgmma plus wait_group 0, alone on the SM: 105 cycles (n64, 32 cycles of math), 137 (n128, 64);
    4 wgmma + wait: 190 / 318.
  - So each drain-to-zero costs about 60-70 cycles that only other warpgroups can hide.
  - A named-barrier ping-pong handoff costs 37-39 cycles (b_pingpong_w2 / w3: 74 / 117 cycles per round).

### 2b. Issue and pipes without tensor work (a_*, x_*)

- **Per SMSP per clock:**
  - FFMA / FMUL / FADD: 1.0.
  - FMNMX, FSEL, LOP3, IMAD, F2FP: 0.5.
  - SHFL: 0.25.
  - LDS.64: 0.125.
  - MUFU.EX2: 4 lanes (16 / clk / SM, exactly).
- **No dual issue:** IPC never exceeds 4 per SM. FFMA + FMNMX 1:1 reaches 3.96; EX2 + FFMA 1:8 reaches 3.81 IPC with
  MUFU at 13.5 lanes/clk.
- **The real mixes alone** (tensor work removed, same warpgroups) reach IPC 3.52-3.57 per SM: 88-89% of the 4 / clk
  issue floor.
  - Forward and pass 0 are MUFU-bound at 91-94% of the MUFU floor.
  - Pass 1 is issue-bound at 88-92%.
- **So 4 issues per SM per clock is not reachable with these mixes**, even with the tensor pipe idle.

### 2c. Tensor and ALU together: interference (i_*, r4 = 21183628 and r7 = 21183994, reproduced to the cycle)

Warpgroup 0 streams SS n128 wgmma at full rate (1024 cycles per step); warpgroups 1-3 run one instruction class sized
to about 768 cycles alone. Cycles per step of the ALU warpgroups:

| class | alone | with tensor stream | slowdown |
|---|---:|---:|---:|
| FFMA | 834 | 1673 | 2.01x |
| LDS.64 + 2 FFMA | 790 | 1540 | 1.95x |
| SHFL | 768 | 1538 | 2.00x |
| FMNMX | 788 | 1292 | 1.64x |
| IMAD | 763 | 1294 | 1.70x |
| MUFU.EX2 | 755 | 1024 | >= 1.36x (ends with the tensor role) |
| FFMA, only 2 ALU warpgroups | 833 | 1657 | 1.99x |
| FFMA next to an SS n64 stream (2x the wgmma instructions) | 834 | 1741 | 2.09x |

- **While the tensor pipe runs at full rate, the other warps of the SM keep about half of their issue capacity.**
  FFMA drops to 0.46 per SMSP per clock; ALU-pipe ops drop to 58% of their pipe rate.
- **Not a scheduling artifact.** Fewer ALU warps and twice the wgmma instructions change nothing, and the tensor role
  itself keeps 1024 cycles per step.
- This is the kappa of section 2: it costs the forward about 830 cycles per step (C_best 2909 vs C_alu 2085).
- With the full mixes, warp specialization shows the split:
  - forward ALU warps: 2085 -> 2909 cycles;
  - the non-MUFU part alone: 2599 with the tensor stream;
  - the MUFU part alone: 2162.

### 2d. Clock and power (`smi_phases.json`, `sustain.jsonl`)

- **Under the power cap.** The real forward and backward (B1 N65536 32/4 loops) and the synthetic kernels with random
  operands all sit at the 700 W cap: 686-691 W, reason 0x4 (SW power cap).
- **nvidia-smi misses the dips.** It reports 1815-1950 MHz, while clock64 / globaltimer inside the kernels gives
  1660-1800 MHz (sustained 3 s).
- **The dip is not only power.** With zero operands at 255-400 W, clock64/ns still reads 1635-1815 MHz while
  nvidia-smi shows 1980: the SM clock dips whenever the tensor cores are busy.
- **Without tensor work** (ALU mixes alone): 375-482 W, 1980 MHz.
- **Consequence for sol4.** sol4's MUFU / issue floors use the nvidia-smi clock sampled during the calibration GEMM.
  That sample itself read 1980 MHz in some jobs and 1800 in others. Under tensor load these floors are 10-17%
  optimistic.

### 2e. Sensitivity of the real structure (s_*, r8 = 21184066, calibration 1512)

Base values (dep structure): forward 62.5%, pass 1 58.2%, pass 0 68.3% of sol4.

| change per warp per step | forward | pass 1 | pass 0 |
|---|---:|---:|---:|
| -50 instructions | 64.1 | 58.8 | 75.4 |
| -100 | 55.6 (schedule changed) | 61.5 | 80.3 |
| -150 | 63.1 | 65.3 | 83.8 |
| +50 | 56.4 | 57.3 | 60.4 |
| +100 | 52.2 | 54.9 | 53.4 |
| 25% of EX2 -> FFMA | 63.1 | 56.7 | 71.1 |
| 50% | 63.3 | 57.3 | 69.4 |
| 100% (no MUFU) | 64.1 | 58.8 | 70.1 |

- **Pass 0:** gains about 0.16 percentage points of sol4 per instruction removed, and still stays below 85% at
  -150 (56% of its loop).
- **Pass 1:** gains about 0.05 points per instruction.
- **Forward:** removing all MUFU gains only 1.6 points; instruction cuts are non-monotonic. In the forward's own order
  (PV -> wait -> fold -> S -> wait -> softmax per warpgroup), the structure plus the coupling binds, not MUFU or issue.
- **No MUFU-reduction lever (e.g. exp2 emulation on the FMA pipe) can bring any of the three near 90% of sol4.**

## 3. Conclusions per kernel

The forward at 58% of sol4 equals 82% of what its mix allows; pass 1 at 42% equals 68%; pass 0 at 70% equals 94%.

- **Pass 1: highest reachable is about 62% of sol4** (ideal 62.0%, interleaved; the real order reaches 61%).
  - The real kernel is at 68% of that ceiling: a 32% gap that is NOT in the step loop's mix.
  - Candidates for the gap:
    - executed instructions above the charged cheapest path (the dV promote / rescale arms, the dS' store block);
    - producer / TMA and LSE / delta loads;
    - the dependence order across its only 2 warpgroups;
    - tiles and tails.
  - The pass-1 helpers' remaining budget is about 0.68 -> 0.9 of 62%, i.e. pass 1 <= ~56% of sol4 (target ~136-139 ms on
    the four shapes vs 179 ms now).
- **Forward: highest reachable is about 70% of sol4** (warp-specialized ideal; the real dependence order reaches 65%).
  - The real kernel is at 82% of the ideal.
  - 90% of the ideal-mix floor means about 63% of sol4: roughly 61 ms vs 66 ms now on the four shapes.
- **Pass 0: highest reachable is about 75% of sol4.**
  - The real kernel is already at 94% of the ideal: the 90% goal is met against its mix.
  - Only a shorter loop (fewer instructions per step, which is a numerics question here) moves the ceiling.
- **Under sol4 as defined, 90% is unreachable for all three.** Even an ideal kernel would need, per warp per step:
  - forward: fewer than about 440 issued instructions and about 60 MUFU, and a different order;
  - pass 1: fewer than about 400 instructions;
  - pass 0: below about 100 instructions, beyond the -150 point of the sweep (83.8%).

  (Section 2 model at 90%: C_alu + kappa x C_tensor <= 1.3 x C_tensor / 0.9.)

## 4. Proposed corrected floors

1. **sol4c (recommended for the gate).** Ideal-mix floor = sol4-v2 floor / ideal fraction:
   - forward 0.699, pass 1 0.620, pass 0 0.745 (`factors.json`, two jobs, per-job spread +-0.002);
   - pass 2 unchanged.
   - `python solcheck/sol4c.py --sol4 <sol4v2.json>`. On the accepted r17_p0s: forward 83.3%, pass 1 68.9%,
     pass 0 94.8% of the ideal-mix floor (`sol4c_r17p0s.md`).
   - For a gate, measure real and ideal in the same job (`run_job.sh` with REAL=1). Their ratio does not depend on the
     job's calibration.
   - The factors belong to the r17_p0s step-loop mix. A candidate that changes its loop's mix must re-measure its own
     ideal (`mix.py` -> `cfgs.py` -> `build_job.sh` -> `run_job.sh`), or its floor would move with its own code.
2. **Repairs to the sol4 model itself** (all measured above):
   - (a) MUFU / issue floors at the effective SM clock under tensor load (clock64/ns, 1.69-1.80 GHz), not the
     nvidia-smi sample (1800-1980): +10-17%.
   - (b) Per-pipe issue rates instead of 4 / clk: ALU-pipe ops at 0.5 per SMSP per clock, SHFL at 0.25. Best measured
     IPC for these mixes is 3.52-3.57.
   - (c) A coupling term: time >= (C_alu + kappa x C_tensor) / f_eff with kappa ≈ 0.55 (0.52-0.62 measured), instead of
     max(tensor, MUFU, issue).
   - (d) The tensor peak calibration varies +-5% between jobs (1381-1513 TFLOP/s for the same GEMM) and sits 17-24%
     below what wgmma sustains at the power cap. Gate on same-job ratios, or calibrate the tensor peak as the median of
     several jobs.
   - With (a)-(c), sol4 predicts the ideal step kernels within 1-5% (forward 2783 vs 2909 cycles, pass 1 2065 vs 2163,
     pass 0 1702 vs 1724, kappa fitted per kernel).

## Limitations

- The ideal kernel covers the steady-state step loop only. Producer / TMA traffic, masked and peeled steps,
  prologue / epilogue, the tile scheduler and causal tails are left out. That makes it an upper bound for a real kernel
  with this mix, so the corrected floors are still optimistic (lower-bound) times.
- Uniform-datapath and control ops are emitted as LOP3 / IMAD filler, which occupies ALU / FMA-pipe slots the real
  uniform ops do not. Effect <= ~1-2 points (about 50 such ops per warp-step in the forward; sweep slope above).
- The free-mode synthetics CSE part of their F2FP into fewer instructions. The loop length is still matched with filler.
- The counts are sol4's cheapest-path counts, a lower bound on what the real kernels execute (pass 1 executes its dV
  promote / rescale arms on 1/8 and 1/2 of the steps).
