# kda-pass1 candidates (pass 1, D128 causal matched)

Helper kda-pass1, branch `kda-sm90-pass1` (rebased on `origin/kda-sm90` c563139 = round 17). Every lever is a
compile-time switch in `csrc/mainloop_bwd_sm90_fp8r.hpp` (+ `fq8_dq_pass2_hd128.h`, `fp11_delta_v6.h`,
`delta_matching_api.cpp` where noted), default OFF: with no switch set the hd128 pass-1 SASS is identical to the
accepted build (checked against frozen r14_h3). A candidate is the same source with its switches' defaults set
(one branch per candidate). Development hooks: `DM_XDEFS` (extra -D for what-if builds) and `DM_XFLAGS` (flag-set
overrides; such a build defines `KDA_FLAGSET_DEV`, which skips the `dm_flagset.h` release check) in `_buildcfg.py`.

Measurement: A = frozen `r17_c2` (pass 1 identical to r14_h3), `benchmarks/ab.py run --grid paper` (B2 N8192 and
B1 N65536 at 16/4 and 32/4), 3 alternating reps, two fresh jobs. Screens: `ab.py --grid one --smoke`, 2 reps, B2
N8192 at 32/4 and 16/4 (noise ~0.2-0.5%). Equivalence: `benchmarks/equivalence.py`, 72 cases per route.
SASS: executed-instruction estimate per warp per m-block step (static loop count with the dV promote arm weighted
1/8, the dV rescale arm 1/2, single-thread arms 1/8).

## READY

### C1: TOPFILL=2 + DVZI + LSEPRE + PV4 + DSDEFER (bitwise)

- Branch `kda-sm90-pass1-c1` (the commit on top of `kda-sm90-pass1` that sets the five defaults; nothing else).
- What it changes (all schedule / layout; same values, same order of arithmetic):
  - TOPFILL=2: the dV window rescale (64 FMUL, every other step) runs right after the S / dP issue, in the bubble
    where the warpgroup waits for its own S gemm, instead of between the dS line and the dV issue.
  - DVZI: the first dV gemm of each promote window runs with scale-d = 0; the promote no longer re-zeroes the
    64-register window accumulator and the window's first rescale is skipped.
  - LSEPRE: the host forms `lse * log2e` (the kernel's exact `__fmul_rn`) once per backward call for the hd128 pass 1
    (`lse.mul(float(M_LOG2E))`); the kernel drops 16 FMUL per thread per step. Pass 0 / pass 2 / d256 unchanged.
  - PV4: the dV smem overlay as float4 slots (16 LDS.128 / STS.128 per promote instead of 64 + 64 scalar).
  - DSDEFER (the M20 design re-implemented on the H1b source): dS' half tile double-buffered (+8 KB smem); the
    elected thread drains store(m-2) before tile m's psi barrier and issues store(m-1) right after it; the two
    per-tile WG-local named barriers on the dS' path are gone (one barrier after the loop for the last tile).
- Bitwise: equivalence vs frozen r14_h3 BITWISE EQUAL on default / kv / head / streams2 (72 cases each, jobs
  21173280 + autocheck); pytest 49 passed / 3 skipped.
- Resources: REG 168 both pass-1 symbols; causal symbol STACK 0 -> 8 (one STL / LDL pair at 0xccc0 / 0xce60,
  outside both step loops: epilogue region), non-causal STACK 8 unchanged; LOCAL 0; no ptxas spill /
  serialization diagnostics. Executed estimate per warp-step (unmasked loop) 734 -> 782 (DSDEFER's double-buffer
  selects and the staging moves add instructions; the gain is latency, not count).
- A/B vs r17_c2, `--grid paper`, 3 reps (jobs 21173276, 21173277; `pass1-aux/runs/full_c1b_{1,2}`):

| shape | job 1 pass 1 A -> B ms | change | job 2 pass 1 A -> B ms | change |
|---|---:|---:|---:|---:|
| B2 H16/4 N8192 | 1.844 -> 1.757 | -4.70% | 1.829 -> 1.744 | -4.66% |
| B2 H32/4 N8192 | 3.799 -> 3.637 | -4.25% | 3.838 -> 3.644 | -5.07% |
| B1 H16/4 N65536 | 60.423 -> 57.862 | -4.24% | 61.324 -> 58.962 | -3.85% |
| B1 H32/4 N65536 | 121.059 -> 115.895 | -4.27% | 123.190 -> 118.129 | -4.11% |
| geomean / sum | **-4.36%** / -4.26% | | **-4.42%** / -4.05% | |

  Pass 2 +0.14% / +0.28% (unchanged kernel); glue +2.8% (+0.09 ms over the four shapes: the host `lse * log2e`
  multiply); e2e fwd+bwd geomean -1.71% / -0.98%; no shape slower beyond its spread. Note: the forward and pass 0
  (byte-identical binaries) read +0.9 .. +1.4% at the two N65536 shapes in every job of every candidate (bwall, nball,
  c1b), never at N8192: an artefact of the run order / clocks around a faster pass 1, not code.
- Status: **READY** (>= 3% on the target kernel in two jobs, bitwise).

## Output-changing (screened against Hao's numerics rule; numerics capture pending)

### N1: C1 + EXPB + PW16 ("nball") / PW32 ("nb32")

- EXPB (pass 0 AND pass 1, so the matched P stays identical between them): the 2^8 offset folded into the per-row bias,
  b = fl(fl(lse * log2e) - 8), P^ = exp2(min(fl(s * dsc - b), 12)): one rounding instead of two, one FADD per element
  less (pass 0 too: pass 0 -0.7%). With LSEPRE the host forms b (`lse.mul(log2e).sub_(8)`).
- PW16 / PW32: the dV promote window 8 -> 16 / 32 m-blocks (fewer promotes; the window accumulates longer in the wgmma
  accumulator).
- nball (C1-era stack TOPFILL=1 + PV4 + DSDEFER + DVZI + LSEPRE + EXPB + PW16), A/B vs r14_h3 `p1acc` = paper set,
  two jobs: pass 1 geomean **-7.43% / -7.14%** (sum -7.18% / -7.25%), pass 0 -0.7%, glue +5.0%, e2e fwd+bwd geomean
  -3.36% / -3.24%. pytest 49 / 3. nb32 screen: -8.96% / -8.60%.
- Status: needs the numerics check (residual non-regression per row, exemption set, determinism, per-case error vs
  fp32 not larger). Rebuild on the C1 stack (TOPFILL=2) pending.

## REJECTED / not useful (each measured; screen numbers are pass 1, B2 N8192 32/4 and 16/4, vs the accepted build)

| variant | what | result | reason |
|---|---|---:|---|
| REDUX=1 | psi amax: per-thread fmaxf tree + redux.sync.max.u32 + 8-slot tree (bitwise) | +3.08% / +2.88% | slower (REDUX latency; the chain interleaves with the dS line) |
| REDUX=2 | per-thread chain + redux.sync (bitwise) | +1.72% / +1.78% | slower |
| REDUX=3 | per-thread tree + shuffles (bitwise) | +0.18% / +0.16% | no gain |
| DVLATE | dV issued next to dK (pre-H1a order) | +3.63% / +3.68% | confirms H1a |
| EARLY=1..5 | S / dP of m+1 issued at the end of step m (5 placements; with and without a liveness kill) | +70 .. +200% | 300-560 B stack, C7511 wgmma serialization: the register budget (dK 64 + dV 64 + tmpK 64 + S 32 + dP 32) does not fit |
| FOLDSPLIT | half the dK fold in the next step's S bubble | +15.9% | spills |
| XPTOP | dS' shuffles / stmatrix in the next step's S bubble | +1.44% / +1.81% | slower |
| UNROLL2 | compile-time Q / Qt stages (unmasked loop in pairs) | +8.9% | spills (50-73 STL/LDL) |
| TOPFILL=1 in the stack | promote also moved into the S bubble | bwall -4.10/-4.24 vs TOPFILL=2 -4.40/-4.70 | the promote is smem-bandwidth bound; moving it does not help |
| HALFPSI + P2A | per-warpgroup psi (WG-local barrier) + pass 2 with two accumulators | pass 1 -2.13 / -2.85%, **pass 2 +12..13%** | the pass-2 consumer costs more than pass 1 gains (also C7515 in the first P2A, fixed) |
| HALFPSI + P2B | per-warpgroup psi + pass 2 halves in key order (one accumulator, a wait per half) | pass 1 -2.18%, pass 2 +10.7% | same |
| HALFPSI + DOST/DOTST 2 | decoupling with 2-stage dO / dOt | pass 1 -0.5% | 2-stage dO costs +1.4% on its own (do2) |
| PP1 / PP2 / PP3 (+HALFPSI, with / without DO2) | ping-pong: one-time WG offset; turn barriers on S/dP; turns on both gemm batches | -1.06 .. +1.0% | no gain: with everything serialized, the lone warp per SMSP is latency-bound |
| DSDEFER alone, PV4 alone | | +0.33% / -0.04% | no gain alone (DSDEFER helps ~2% inside the C1 stack) |
| POFF | producer warps 1..3 do the dV window promote | pending | |

Timing-only ablations (KDA_ABL, garbage outputs) of the accepted step, pass 1 change when the piece is removed: exp line
-9.5%, dS' transpose + stmatrix + TMA store -11%, dK fold -8.5%, dV promote (1 step in 8) -11%, psi amax + barrier
-9.3%, e4m3 conversions -5.3%, dV rescale -2%, dS line -2%; S + dP gemms -21%, dV + dK gemms -9.7%, all gemms -30%;
all vector work -63%. The pieces are nearly additive: vector and tensor work do not overlap. The remaining gemm + TMA
skeleton (~37% of the step, ~1330 cycles) is close to a shared-memory bandwidth floor that sol4 does not model: per
CTA step ~176 KB of smem traffic (S / dP SS wgmma read K/V + Q/dO per warpgroup 64 KB, dV / dK RS read dOt / Qt 32 KB,
TMA writes 32.5 KB, dS' stmatrix + store 16 KB, promote 16 KB amortized, LSE / dPsum loads) at 128 B/clk = ~1375 cycles,
vs the calibrated tensor floor ~1580 cycles. The full screen log is `pass1-aux/screens_table.md`.
