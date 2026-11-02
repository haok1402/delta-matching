# kda-p1fresh candidates (pass 1 at head dim 128 from scratch: decompositions that change the per-element work)

Helper kda-p1fresh, branch `kda-sm90-p1fresh` (on `origin/kda-sm90` 280ec45 = r18_c4 accepted, round 19 started).
Brief: explore decompositions of pass 1 (dK, dV, dS') that change the per-element work, each as a minimal prototype
timed against the accepted pass 1 (frozen `r18_c4`); the bar is a prototype that beats it by **10%**; drop the
designs that cannot.

**Result: no READY candidate. No decomposition reaches the 10% bar.** The only prototype bound above 10% (FR3: all
dK / dV work removed from the compute warpgroups, -12..-15%) belongs to the full role split. Built as a real
four-warpgroup prototype, that split is **8-10% slower** than r18_c4. Every other lever's upper bound is below
10% before its costs are paid.

## Method

- **Switches.** Every probe is a compile-time switch in `csrc/mainloop_bwd_sm90_fp8r.hpp` (and
  `csrc/flash_bwd_kernel_sm90_fp8r.h` for the fourth warpgroup): `KDA_FR` (bit mask, default 0), `KDA_FR_CREG` /
  `KDA_FR_PREG` (register split of the prototype), `KDA_FR_MMAREG` (compile-only register-demand probe). They are
  built with `DM_XDEFS`.
- **Default build.** With no switch set, the pass-1 SASS of both instantiations is identical to frozen r18_c4 (md5 of
  the address-stripped functions). The base build also screens at -0.60% / -0.04%, which is noise.
- **Timing.** `benchmarks/ab.py run --grid one --smoke`, 2 alternating reps, A = frozen r18_c4 in the same job.
  Shapes are B2 N8192 and B1 N65536 causal matched at Hq/Hk 32/4 and 16/4. Numbers are the pass-1 median change of
  B vs A; rep spreads are <= 0.9% except where noted.
- **Tools.** Kept outside the repo, in `kda-work/p1fresh-aux/`:
  - `jobs/{build,tu,screen,screen2}.sh`
  - `tools/qstat_bytes.py`
  - `builds/<name>/` (build.log, resource.txt, all.sass, root)
  - `tu/<name>/`
  - `runs/`
  - `jobs/logs/`
- **Outputs.** All probes are TIMING-ONLY (wrong outputs) unless stated otherwise.

| switch | what the compute warpgroups no longer do (or do differently) |
|---|---|
| FR1 DKOUT | no dK gemm, no fold; the 8 dS'-code A-fragment words go to an smem hand-off tile (2 STS.128) |
| FR2 DVOUT | no dV gemm, no rescale, no promote; the 8 P-code words go to an smem hand-off tile (2 STS.128) |
| FR3 = 1 + 2 | both: the compute warpgroups keep only S, dP, exp, dS, psi, the codes and the pass-2 transpose / store |
| FR4 NOCVDS | no psi multiply and no e4m3 conversion for dS' (raw fp32 bits feed the A operand and the transpose) |
| FR12 = 4 + 8 | also no e4m3 conversion for P |
| FR16 WGPSI | psi per warpgroup (WG-local barrier): the decoupling a 64-key CTA would get |
| FR32 NOFOLD | no dK fold (the dK gemm stays) |
| FR67 PHANTOM | FR3 + a fourth warpgroup (the gemm role). It reads the codes from double-buffered hand-off tiles (mbarriers fullP / fullD / empty) and issues the dV and dK wgmmas for all 128 keys (8 QGMMA per step). It also runs the dV rescale (128 FMUL, every second step), the dV promote (128 values, every 8th step) and the dK fold (128 FFMA per step). Register split 24 / 120 / 120 / 248 |
| FR66 | the same with a dV-only gemm warpgroup (the compute warpgroups keep dK): the register-feasible-looking split |

## Measurements (pass-1 change vs r18_c4)

| probe | job 21193077: 8K 32/4, 8K 16/4 | job 21193349: 8K 32/4, 8K 16/4, 64K 32/4, 64K 16/4 | job 21193848 (same four) | resources |
|---|---|---|---|---|
| base (no switch) | -0.60 / -0.04 | | | SASS identical to r18_c4 |
| FR1 DKOUT | -8.01 / -7.25 | -7.32 / -7.16 / -8.38 / -7.94 | | REG 168, STACK 8 |
| FR2 DVOUT | -7.26 / -6.51 | -7.66 / -6.63 / -8.13 / -7.51 | job 21194215: -7.24 / -6.46 / -7.91 / -7.64 | REG 168, STACK 8 |
| **FR3** | -12.28 / -11.88 | **-13.87 / -11.83 / -14.60 / -14.30** | -13.57 / -11.34 / -15.29 / -14.94 | REG 168, STACK 8 |
| FR4 NOCVDS | -0.43 / -1.18 | | | REG 168, STACK 0 |
| FR12 | -1.25 / -1.72 | | | REG 168, STACK 0 |
| FR16 WGPSI | -1.66 / -1.79 | | | REG 168, STACK 0 |
| **FR32 NOFOLD** | -9.73 / -9.65 | **-10.61 / -9.73 / -10.24 / -10.46** | | REG 168, STACK 8 |
| **FR67 PHANTOM** (full role split) | | | **+9.89 / +10.22 / +8.38 / +8.34** | 512 threads; REG 128 at launch, STACK 8-16 (16 B spill), no C75xx, 24 QGMMA |
| FR66d (dV-only gemm warpgroup) | | | job 21194215: **+22.38 / +22.91 / +21.20 / +21.81** | compute 152 / gemm 184: STACK 104, 196 B spill; 160 / 168: C7512 wgmma serialization |

Register demand (compile-only, `KDA_FR_MMAREG`, TU jobs 21192867, 21193750, 21193921 and the 160 / 168 and 152 / 184 splits):
- An FR3 compute warpgroup fits in 112 registers (12 B spill, the same 12 B as at 240). At 96 it spills 188 B.
- An FR1 compute warpgroup (which keeps dV) fits at 168-176. At 160 it spills 24 B.
- An FR2 compute warpgroup (which keeps dK) fits at 168-176.

## REJECTED (one entry per design; the lever and its variants)

### Lever 1: query-block-stationary pass 1 with deterministic cross-CTA dK / dV reduction

**Status: REJECTED, bytes-bound.**

- **Design.** A CTA owns R query rows (all of pass 0's per-row state) and loops over the key blocks. Every (R-row
  chunk, 128-key block) pair leaves an fp32 dK + dV partial of 128 keys x 128 d x 2 = 128 KB. An ordered reduction
  (no atomics) writes each partial once and reads it once.
- **Measurement** (`tools/qstat_bytes.py`, job 21193349). One device copy of exactly those bytes per shape, the
  cheapest possible write + read, at about 3 TB/s. Summed over the four shapes:

| R (rows per CTA) | partial traffic alone | vs accepted pass 1 total |
|---:|---:|---|
| 64 | 1127 ms | |
| 256 | 282 ms | (4 consumer warpgroups, the pass-0 W4 shape) |
| 512 | 141 ms | |
| (r18_c4 pass 1, all work) | 172.7 ms | |

  The 10% bar is 155 ms total. At R = 512 the reduction traffic alone is 141 ms, before any compute, and 512 rows
  per CTA do not fit in registers anyway.
- **Variant 2** (codes through HBM). The query-stationary kernel writes the P codes (and the dS' codes it already
  writes) and a gemm-only dK / dV kernel reads them back. The extra bytes are a P write + P read + dS' read, about
  3 x 36.7 ms = 110 ms of bytes floor (pass 1's bytes floor is 36.7 ms at 1 B per element). The second kernel also
  carries half of pass 1's tensor floor (37.5 ms) and repeats no ALU work. That is above 155 ms before the
  per-element work is counted.
- **Variant 3** (cluster / DSMEM reduction across the CTAs of one key block). It moves the same 128 KB per tile
  per step over the SM-to-SM network, which is slower than local smem.
- **Note.** Family A / B of round 18's structural worker (pass 0 and pass 1 sharing work) are the fused forms of
  this lever and are already rejected.

### Lever 2: role split between warpgroups (ALU stream vs tensor stream)

**Status: REJECTED, measured.**

| variant | bound (compute warpgroups without the moved work) | real prototype | registers |
|---|---|---|---|
| (a) dK warpgroup: dK gemm + fold moved out | FR1 -7.2 .. -8.4% | not built: below 10% as an upper bound (and FR66d shows what a fourth warpgroup next to 168-register compute warpgroups costs) | the dK warpgroup needs dK 128 + temp >= 32 for 128 keys; with compute warpgroups at 160 (24 B spill) the sum is >= 24 + 320 + 172 = 516 |
| (b) dV warpgroup: dV gemm + rescale + promote moved out (p1redesign's DVWG; its ABLDV bound on r17_p1 was -8.8%) | FR2 -6.5 .. -8.1% | **FR66d: +21.2 .. +22.9% (slower)**, spilling at the only split without wgmma serialization | 24 + 2 x 168 + 168 = 528 > 512: at 160 / 168 ptxas serializes wgmma (C7512); at 152 / 184 the compute warpgroups spill 196 B |
| (c) full split: compute warpgroups keep only S / dP / exp / dS / psi / codes; a gemm warpgroup does dV, dK, fold, rescale, promote | **FR3 -11.3 .. -15.3%** | **FR67 PHANTOM: +8.3 .. +10.2% (slower)** | the prototype undercounts: it reuses one 64-register dV accumulator and one dK temp. A real gemm side needs dV 128 + dK 128 + temp >= 32 per thread for 128 keys, against 264 left after the producer (24) and two 112-register compute warpgroups |

Why the full split loses 22-25 points against its own bound:
- The bound deletes half of the tensor work: the dV / dK wgmmas run nowhere. The prototype runs them concurrently
  in the fourth warpgroup.
- That reproduces solcheck's coupling (C_alu + 0.55 C_tensor). With the tensor pipe busy, the compute warps keep
  about half their issue rate.
- On top of that come:
  - the gemm warpgroup's own fold, rescale and promote stream: one warp per SMSP, latency-bound;
  - the hand-off (4 STS.128 + 2 arrives per compute thread per step, 4 LDS.128 + waits on the gemm side);
  - the extra warps' issue slots.
- solcheck's ideal-mix kernels already showed that warp specialization does not beat the dependent order (spec 61.4%
  vs dep 61.7% of sol4). The real prototype confirms it with the real dependence chain and is clearly worse.
- This also closes the "64-key CTA + 2 compute warpgroups ping-ponging m-blocks + 1 gemm warpgroup" form (the only
  register-feasible full split, at 24 + 2 x 112 + 208 = 456). It is the same split with the same coupling. It
  also adds per-64 x 64 psi (pass 2 +7..9%, round 17) and twice the Q / dO / Qt / dOt traffic per key.

### Lever 3: key block size with a different psi tile granularity (named numerics change)

**Status: REJECTED, bound below 10% or infeasible.**

- **(a) 64-key CTAs** (1 consumer warpgroup at 232 + 24, 2 CTAs / SM, psi per 64 x 64).
  - The gain is the decoupling of the two 64-key halves. The bound is FR16 (WG-local psi, no CTA-wide barrier):
    -1.7 / -1.8%.
  - Against it: per-64 psi in pass 2 (+7.4 .. +8.6% of pass 2 measured in round 17, about +1.7% of pass 1's time)
    and twice the Q / dO / Qt / dOt loads per key.
- **(b) Coarser psi** (one psi per 128 rows x 128 keys, so one fold per two m-blocks).
  - The bound for removing the whole fold is FR32: -9.7 .. -10.6% (mean -10.3%). Halving it is worth about -5%.
  - A real design must hold the first tile's fp32 dS (32 registers) across the second tile's S / dP: 224 + addressing
    > 240, the spilling FOLDSPLIT / deferred-fold family.
  - It also coarsens the dS' codes of half the tiles (more e4m3 underflow), the kind of change that failed
    residual-v4 for the forward's shared basis.
  - Removing the fold without a coarser psi (running psi with rescale-on-change, DKACC) was neutral or failed
    residual-v4 in round 18.
- **(c) Larger key blocks** (192 or 256 keys per CTA, 3-4 consumer warpgroups).
  - 3 warpgroups need <= (65536 - 3072) / 384 = 162 registers per thread; the 64-key state is S / dP 64 + dK 64 +
    dV 64 (+ temp 64) >= 192.
  - 4 warpgroups need <= 122.
  - Only FR3-type compute warpgroups (112) fit, which needs the role split of lever 2 (rejected).
  - A larger key block saves only Q / dO / Qt / dOt traffic per key, which is not the bound: pass 1's bytes floor is
    36.7 of 172.7 ms.

### Lever 4: dS' codes in a narrower form that removes conversions

**Status: REJECTED, bound below 10%.**

- **(a) FR4.** No psi multiply and no e4m3 conversion for dS': -0.4 / -1.2%.
- **(b) FR12.** No conversion for dS' or P: -1.3 / -1.7%.
- **Earlier evidence.** DSNAT (p1redesign: dS' stored in the dK A-fragment layout, transpose moved to pass 2) was 0%
  on pass 1 and +1.9% on pass 2.
- **No narrower operand exists on sm_90.** There is no fp4 / e2m1 wgmma on sm_90; e4m3 is already the narrowest
  operand.

## Final summary

- Every lever in the brief was tried with at least two designed variants:
  - Lever 1: three variants, bytes-bound.
  - Lever 2: three variants. Two were bounded below 10%; the third was built as a real four-warpgroup prototype
    (+8..10%) plus the register-feasibility builds.
  - Lever 3: three variants (64-key decoupling bound, coarser-psi fold bound, larger-CTA register arithmetic).
  - Lever 4: two variants.
- None beats r18_c4's pass 1 by 10%. Nothing is READY.
- The measurements leave one structural fact for whoever continues.
  - Pass 1's removable per-element work sits in the dK / dV half of the step:
    - fold: -10%;
    - dV or dK path: -7..-8%;
    - both: -12..-15%.
  - That work cannot leave the compute warpgroups without either running concurrently on the same SM (coupling:
    measured +8..10% for the full split) or lacking a register home (dK 128 + dV 128 + temp per 128 keys).
  - The fold itself has no cheaper exact form on sm_90 (round 18).
  - So the remaining pass-1 gap (69% of sol4c) is fixed by the numerics contract of the per-tile psi basis together
    with the register file, not by the decomposition.
- The switches stay in the source, default off and compiled out, as a record. The no-switch build is the accepted
  kernel. KDA_FR builds are timing-only (wrong outputs); do not gate them.
