# kda-fuse candidates (branch kda-sm90-fuse)

Helper scope: cross-kernel savings in the matched backward (pass 0 = matched delta, pass 1 = dK / dV / dS', pass 2 = dQ):
remove duplicated S = Q K^T / dP = dO V^T work, fuse passes, change the launch / grid structure. Every variant is measured
on the WHOLE backward (backward-only wall time on a retained graph, plus per-kernel CUPTI times and inter-kernel idle
gaps, `benchmarks/fuse/bwd_timeline.py`) on the four acceptance shapes (D128 causal matched, Hk 4: B2 N8192 and B1 N65536
at Hq 16 and 32). A = the accepted build `.local/frozen/r17_p1` (origin/kda-sm90 csrc is unchanged since). A/B jobs
alternate the order of the roots per repetition, fresh process per run, 3 repetitions (`benchmarks/fuse/ab.sh`,
`summ.py`); bitwise = `benchmarks/equivalence.py` default route, 72 cases, vs `.local/equiv/base3_default.pt`.

All source switches below are `KDA_*` macros, default 0, passed only through `KDA_DEFS` in screening builds
(`_buildcfg.defines()` appends them); with the defaults the build is the accepted kernel set.

**Status: no candidate READY. Every lever has at least two designed variants measured or priced by measured numbers;
the measured ceilings of the cross-kernel levers are below the 3% (or 1%-bitwise) bars. Summary at the end.**

## Baseline structure of the backward (job 21176369, `results/fuse/t0`)

| shape | bwd wall ms | pass 0 | pass 1 | pass 2 | glue + producers | idle between kernels | launches |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2 N8192 16/4 | 2.895 | 0.562 (19.4%) | 1.745 (60.3%) | 0.431 (14.9%) | 0.157 | 0.095 ms | 7 |
| B2 N8192 32/4 | 6.000 | 1.171 (19.5%) | 3.720 (62.0%) | 0.851 (14.2%) | 0.260 | 0.008 ms | 7 |
| B1 N65536 16/4 | 88.706 | 18.643 (21.0%) | 56.942 (64.2%) | 12.073 (13.6%) | 1.143 | 0.094 ms | 55 (16 head calls) |
| B1 N65536 32/4 | 179.842 | 37.049 (20.6%) | 114.058 (63.4%) | 24.168 (13.4%) | 2.105 | 0.179 ms | 103 (32 head calls) |

Back-to-back kernels on the stream leave ~1 us gaps (pass 0 -> 1 -> 2), 25-60 us per backward at 64K over all head
calls: the inter-kernel idle time is <= 0.2% of the backward. Pass 2 does not recompute S (it reads pass 1's dS' tiles and
the K twin); the duplicated GEMM work is pass 0's S and dP (pass 1 recomputes both, bit for bit).
Split-route glue at 64K: 3 `cat` copies of the per-head dq / dk / dv (0.55 / 1.10 ms) + the GQA fold (0.35 / 0.49 ms) =
1.0-0.9% of the 64K backward (not this helper's lever; noted for the integrator: writing each head call's outputs into
strided slices of one (B, N, H, D) tensor would remove the cats).

## Lever 1: pass 0 fused into pass 1 (two-phase)

### F1a PDL chain (two-phase at grid granularity): REJECTED (no gain)
- Commit: this branch, `KDA_PDL=1` (`csrc/fp11_delta_v6.h`, `flash_bwd_kernel_sm90_fp8r.h`, `flash_bwd_fp8v10_launch.h`,
  `fq8_dq_pass2_hd128.h`). Pass 0 triggers `griddepcontrol.launch_dependents` after its pipeline init; pass 1 is
  launched with programmatic stream serialization, waits (`griddepcontrol.wait`) before any global access and triggers
  pass 2; pass 2 launched the same way, waits after its barrier init. This is the two-phase pass 0 -> pass 1 schedule with
  a grid-wide phase boundary (the dependent CTAs are resident and past their prologue when the primary drains); a fused
  two-phase kernel with a grid barrier cannot do better than this, because no pass-1 work can start before all of delta
  exists (every key-block CTA needs delta for all rows below its diagonal, the first of which pass 0's LPT order computes
  last).
- Bitwise: yes (72 cases, job 21177232). Resources: unchanged (REG 168 / STACK 0 / LOCAL 0 on the pass-1 / pass-0 /
  pass-2 symbols; no diagnostics).
- A/B (job 21177232, 3 reps): backward wall +0.11% geomean (16/4 8K -0.09%, 32/4 8K +0.10%, 16/4 64K +0.25%, 32/4 64K
  +0.17%), all within spread. The trace confirms PDL is active (the pass0->pass1 and pass1->pass2 gaps vanish, idle at
  64K 121 -> 37 us / 156 -> 66 us), but the gaps were ~1 us each. (The per-kernel pass-2 time reads +7% under PDL because
  its CTAs are resident while waiting; wall time is the metric.)
### F1b pass 1 forms the matched delta itself (row wavefront, no pass-0 recompute): REJECTED (measured cost + infeasible at 64K)
- Design: run pass 1's key-block CTAs of a head in a skewed wavefront (CTA j starts at row step 2j), so all CTAs of a
  head process row block m together; each publishes its per-(row, n-block) partial of sum_j P dP, a fixed-order
  cross-CTA reduction gives delta(m), and the CTA finishes dS for row m with the P / dP it already holds: pass 0
  disappears (19-21% of the backward).
- What-if build `KDA_P1_WI_ROWSUM=1` (timing only, outputs unchanged): pass 1 computes the per-row partial sums of its
  tile every step (fmaf chain over the thread's keys, 3 xor shuffles, 4 warps through parity-buffered smem, a store) —
  only the arithmetic, no cross-CTA exchange, in pass 1's own (cheaper, NOT pass 0's) order. Job 21177232: pass 1
  +17.3% geomean (+19.2% / +17.6% / +15.6% / +16.9%), backward wall +10.8%. That is +0.33 ms per 8K 16/4 call against
  pass 0's 0.56 ms, before (a) pass 0's exact fmaf chain order, which needs the P and dP*dp_g values gathered through
  smem (~64 KB per CTA step; pass 1's smem has 60 KB free and its register file is full at 24 / 240), (b) the per-row-step
  rendezvous of up to 64 CTAs (~1 us L2 round trip in a ~2 us step, or a one-step lag buffer that does not fit), and
  (c) the wavefront's loss of LPT balance.
- Structural: the wavefront needs every key-block CTA of a row co-resident. At N65536 a row spans 512 key blocks (> 132
  SMs), so the design cannot run on half of the acceptance set. Non-bitwise relaxations (pass 1's own reduction order)
  do not lift that limit.
- Resources of the what-if: REG 168, STACK 0, LOCAL 0 (the probe adds 4 KB smem).

## Lever 2: pass 0 stores what pass 1 needs instead of recomputing
Priced from measured numbers (no build): pass 1's dS line is `P_fp32 * (dP_fp32 * descale - delta * k8s)`, so a bitwise
handoff needs both fp32 fragments; visited tile pairs from the causal geometry; calibrated copy bandwidth 3026 GB/s
(round-15 / 16 calibration); pass 0 / pass 1 times of job 21176369.

| shape | visited pairs | F2a fp32 P + dP (bitwise), write + read | F2b fp32 dP only (bitwise) | F2c e4m3 P + dP (not bitwise) | pass 1 + pass 0 today |
|---|---:|---:|---:|---:|---:|
| B2 N8192 16/4 | 1.09 G | 5.77 ms | 2.88 ms | 1.44 ms | 1.75 + 0.56 ms |
| B2 N8192 32/4 | 2.18 G | 11.53 ms | 5.77 ms | 2.88 ms | 3.72 + 1.17 ms |
| B1 N65536 16/4 | 34.4 G | 182.0 ms | 91.0 ms | 45.5 ms | 56.9 + 18.6 ms |
| B1 N65536 32/4 | 68.9 G | 364.0 ms | 182.0 ms | 91.0 ms | 114.1 + 37.1 ms |

Pass 0 still runs in all three (it produces delta), so the saving is at most pass 1's S / dP GEMMs + mask / exp2 (less than
all of pass 1). F2a / F2b cost more than the WHOLE of pass 1 + pass 0; F2c (also a numerics change) costs 82% of pass 1
alone while pass 1 would keep its dV / dK GEMMs, dS, psi, conversions and dS' store. REJECTED (all three).

## Lever 3: fused pass 1 + pass 2

### F3a dS' handed to pass 2 through L2 (ceiling what-if): REJECTED (ceiling below the bar)
- What-if `KDA_P2_WI_L2=1` (timing only, numerics wrong): every pass-2 CTA reads the 64 dS' tiles of one row block
  (512 KB, L2-resident) instead of its own, i.e. pass 2 with its whole dS' stream from L2: the best case of any fusion that
  consumes dS' before it leaves L2. Job 21177066: pass 2 -11.8% geomean (-12.8 / -10.0 / -11.9 / -12.5%), i.e. pass 2 is
  not DRAM-limited even in principle (its per-CTA serial chain dominates); the backward ceiling is ~1.6%. A real fusion
  would also need dS' rows to complete in lockstep (the row wavefront of F1b, with its co-residency limit at 64K) or a
  per-head interleave (F3b), and pass 2 cannot co-reside with pass 1 (pass 1's 384 x 168 registers fill the register file).
### F3b per-head pass 1 / pass 2 interleave on streams (dS' consumed right after its head's pass 1): REJECTED
- Existing bitwise route knobs, no build: `DM_BWD_HEAD_GROUPS=head|kv` x `DM_BWD_STREAMS=1|2|4` at 8K (job 21176458, 2
  reps): best is kv / 4 streams at 3.48 / 6.83 ms vs 2.90 / 6.00 ms unsplit (+20% / +14%); per-head groups +35..+170%.
  The per-group slicing copies (0.7-3.6 ms of glue) and per-group tails dominate; pass 2 per kv group is slower, not
  faster (0.505 vs 0.430 ms).
### F3c dQ accumulated inside pass 1 (no dS' workspace): REJECTED (analytic)
- Deterministic ordered fp32 dQ read-modify-write per (row block, key block) step: 2 x 32 KB per 64 x 128 tile versus 8
  KB written + 8 KB read for dS' today (4x the bytes; 132 SMs x 64 KB per ~2 us step = 4.2 TB/s, above HBM), plus a
  per-row ordering chain across key-block CTAs and a different dQ accumulation (not bitwise; the windowed psi
  promotion of pass 2 cannot be reproduced). Not built.

## Lever 4: launch / grid structure of the three passes

### F4a P2LPT: pass-2 grid (bh, m-block), longest rows of all heads first: REJECTED (slower)
- `KDA_P2_LPT=1` (`fq8_dq_pass2_hd128.h`); motivated by a dispatch model (`benchmarks/fuse/tailsim.py`: pass-2 dispatch
  efficiency 94.5% / 97.2% at 8K vs 98.6-99.9% elsewhere). Bitwise: yes (72 cases, job 21177066). Resources unchanged
  (REG 114 / 116, STACK 0, LOCAL 0).
- A/B (job 21177066): pass 2 +8.2% / +7.6% at 8K, +1.3% / -0.0% at 64K (head-major order keeps one head's K twin and dS'
  rows together in L2; a bandwidth-bound kernel's tail costs less than the model's equal-speed assumption), backward wall
  +0.51% geomean.
### F4b PDL chain: see F1a (bitwise, +0.11%, no gain).
### F4c cross-group concurrency on the 64K split route: REJECTED (no gain)
- `DM_BWD_STREAMS=2 / 3`, `DM_BWD_WS_LIMIT_GB=16` (kv groups of 4 heads), kv + 2 streams (job 21176369, one rep each):
  16/4 64K 88.1-89.1 ms and 32/4 64K 178.1-179.5 ms vs 88.7 / 179.8 ms default: <= -0.7%, within run-to-run spread;
  modelled tails of the per-head kernels (pass 0 97.4%, pass 1 / pass 2 98.6% dispatch efficiency) bound it to ~1.5%.
### F4d persistent pass 1: not built (ceiling below the bar)
- Pass-1 dispatch efficiency is 99.9% (8K) / 98.6% (64K per head) in the model; the per-CTA prologue + epilogue + exit
  (~3400 cycles, round-2 profile) is ~1.4% of an average 8K CTA (65 steps x ~3850 cycles) and ~0.2% at 64K: a persistent
  pass 1 that hid all of it would gain <= ~1.4% / 0.2% of pass 1 (geomean < 1%), below the 1% stacking bar.

## Summary

| id | lever | change | bitwise | result (whole backward, geomean over the 4 shapes) | status |
|---|---|---|---|---|---|
| F1a / F4b | pass0->pass1 two-phase, launch | PDL chain pass 0 -> 1 -> 2 | yes | wall +0.11% (gaps were ~1 us) | REJECTED |
| F1b | pass 0 fused into pass 1 | delta from pass 1's own P / dP (row wavefront) | (needs smem gather) | row-sum work alone: pass 1 +17.3%, wall +10.8%; infeasible at 64K (512 key blocks / row > 132 SMs) | REJECTED |
| F2a-c | pass 0 stores for pass 1 | fp32 P+dP / fp32 dP / e4m3 P+dP | a, b yes | handoff bytes 5.8 / 2.9 / 1.4 ms vs pass 1 1.75 ms (8K 16/4) | REJECTED |
| F3a | fused pass 1 + pass 2 | dS' via L2 (ceiling what-if) | n/a | pass 2 -11.8% at best (~1.6% of bwd) | REJECTED |
| F3b | fused pass 1 + pass 2 | per-head / kv interleave on streams | yes | +14..+170% at 8K (best: kv groups, 4 streams) | REJECTED |
| F3c | fused pass 1 + pass 2 | dQ in pass 1 | no | 4x the dS' bytes, above HBM bandwidth | REJECTED |
| F4a | grid structure | pass-2 global LPT order | yes | pass 2 +4.2% (8K +8%), wall +0.5% | REJECTED |
| F4c | concurrency | 64K split route on 2-3 streams / kv groups | yes | <= -0.7% (noise) | REJECTED |
| F4d | persistent pass 1 | (not built) | yes | ceiling < 1% of pass 1 | REJECTED |

Reading: the three passes already run back to back with ~1 us gaps and 97-100% dispatch efficiency, so launch / grid /
concurrency changes have < 1% to recover. The duplicated S / dP recompute (pass 0 = 19-21% of the backward) cannot be
removed: storing the fragments costs more HBM time than the whole of pass 1, and producing delta inside pass 1 needs every
key-block CTA of a row co-resident (impossible at 64K) plus +17% of pass-1 issue work for the row sums alone. The
cross-kernel savings left are in-kernel ones: pass 0's own issue cost (the pass-0 helper's numerics arms and the P0ALL
all-heads call at 64K) and the 64K split-route output concatenation (~0.6% of the 64K backward, glue).
