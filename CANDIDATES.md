# kda-p1split candidates (pass 1 at head dim 128: breaking the register wall)

Helper kda-p1split, branch `kda-sm90-p1split`. Pass 1 here means dK, dV and dS' at D128, causal, matched.

- **Base.** The branch is rebased onto `origin/kda-sm90` 70b482e (r17_pk, with EXPBX64 accepted).
- **Measurements.** All of them were taken on the 19c238b base against frozen `r17_p0s`, whose pass 1 predates
  EXPBX64. The conclusions are structural and carry over.
- **Exp line.** The PIPE path's exp line is still the pre-EXPBX certified line. It would need EXPBX64's
  per-warpgroup line to be bitwise against r17_pk. It is rejected, so it was not ported.

Every design is a compile-time switch in `csrc/mainloop_bwd_sm90_fp8r.hpp`, default OFF. `DM_XDEFS` (extra -D
defines, ported into `delta_matching/_buildcfg.py` and stamped into the build info) builds a variant.

With no switch set, the build is the accepted kernel. Evidence: the SASS of both pass-1 instantiations is identical
(md5 of the address-stripped Fp8R functions) to the unmodified source's:
- on 19c238b: TU jobs 21182589 and 21184018;
- after the rebase, against 70b482e's own source: TU jobs 21184091 and 21184092.

Tools are in `kda-work/p1split-aux/` (outside the repo):
- `jobs/tu.sh`: a single-TU compile with ptxas -v and the C75xx diagnostics, plus SASS.
- `jobs/build.sh`: a full build from a private snapshot of the worktree.
- `jobs/screen.sh`: `ab.py --grid one --smoke`, 2 alternating reps, B2 N8192 causal matched at Hq/Hk 32/4 and 16/4. A is
  frozen `r17_p0s`, run in the same job.
- `jobs/eq.sh`: the 72-case default-route equivalence against r17_p0s.
- `sassfn.py` and `loopspill.py`: opcode and spill counts per function and per loop.
- Logs are in `jobs/logs/`, builds in `builds/<name>/` (`resource.txt`, `diag.txt`, `build.log`), and runs in `runs/`.

Numbers are the pass-1 median change of B vs A at 32/4 and 16/4. The A/B rep spreads are ≤ 1.1% except in one job
(3.2%, noted).

**Result: no READY candidate.** No variant of any of the five levers comes within reach of a 10% pass-1 gain. The best
real design is slower than the accepted kernel. The best timing-only bound is −8.4%: delete the dK fold entirely,
which no real design can do.

The new finding that closes the direction: **freeing registers for a second S / dP fragment does not buy tensor /
vector overlap in this kernel.** Two schedules were given the same work and the same register headroom (64 freed
registers per thread):

| schedule | pass 1 change (32/4, 16/4) |
|---|---:|
| non-pipelined (accepted order) | −7.71% / −7.22% |
| software-pipelined: the next tile's S and dP in flight under this tile's psi / dS' / dV / dK work | −4.05% / −3.51% |

The pipelined schedule is **3.5-4% slower**, and it is correct: the full-numerics build of the same schedule is bitwise
equal on all 72 cases.

## What bounds the step (measured here, extending p1redesign's notes)

- **The accepted unmasked step loop is issue-bound.**
  - It has 1032 SASS instructions per warp per step, against 12 QGMMA.
  - Two consumer warps share each SMSP, which gives about 2064 issue slots per step.
  - At B2 N8192 32/4 the measured step is about 3430 cycles: 266,240 CTA-steps over 132 SMs at 3.49 ms and 1.98 GHz.
    That is about 60% issue utilisation.
  - The calibrated tensor time is about 1580 cycles per step.
- **Moving tensor work under the vector stream does not shorten the step**, because the vector stream itself is the
  long pole (PIPE result above). Removing instructions on the critical tail does shorten it: deleting the 64-FFMA
  dK fold alone gives −8%.
- **Shared memory cannot hold the accumulators.**
  - A per-step fold of the 128 × 128 fp32 dK through shared memory is 128 KB of LDS + STS per CTA per step, on top of
    about 96 KB of wgmma operand reads.
  - At about 128 B/clk that is a hard +26-27%.
- **Registers.** With dK in registers, a pipelined step needs dK 64 + dV 64 + the next S / dP 64 + 24 operand codes,
  plus about 25 addressing and state registers. That is more than 240.
- **ptxas constraints** (adds to p1redesign's C7511 / C7515):
  - **C7513.** Copying a register into a wgmma A operand serializes every wgmma (C7513), even after a full `wait<0>`
    and even outside the stage. This was bisected with 4 compile variants. The likely cause is that the register
    allocator places the loop-carried copy inside the next iteration's stage.
  - **Workaround.** Converting the next tile's P codes straight into the dV A-operand registers right after dV
    retires (`wait<1>`) is legal and gives batched wgmma.
  - **Unrolling by two.** Ping-pong code buffers in a two-way unrolled loop made the step lambda non-inlined, with a
    1.9 KB stack.
  - **In-place rescale.** An in-place dK rescale inside an open stage (RIP within PIPE) did not serialize.

## REJECTED (prototype measured)

| # | design | lever | build / job | bitwise | pass 1 (32/4, 16/4) | resources | reason |
|---|---|---|---|---|---:|---|---|
| C1 | **DKSM**: the persistent dK accumulator in shared memory (64 KB fp32, float4 slot(j, thread), LDS.128 + STS.128 fold, same `fmaf` per element). Frees 64 registers per thread. Smem 160 → 224 KB | smem accumulator | `dksm`, jobs 21182177 and 21182813 | **yes** (72 cases, job 21183356) | **+26.39% / +27.03%**; second job +26.92% / +26.88% | REG 168, STACK 8 (8 B producer spill), no C75xx | 128 KB of shared-memory traffic per CTA step |
| C2 | **PIPE + DKSM + DKH**: the software-pipelined step (iteration m issues S(m+1); C(m) psi / conversion; dP(m+1), dV(m), dK(m); then the dS' transpose; B(m+1) exp / dS line under dV / dK; P codes converted into the dV operand after dV retires). The producer is reordered: Q(m+1), dO(m+1), dOt(m), Qt(m+1). The dK gemm runs in two N = 64 halves (32-register temps) | smem accumulator + second fragment | `pipe_dkh`, job 21182813 | **yes** (72 cases, job 21183356) | **+20.47% / +20.32%** | REG 168, STACK 16 (3 LDL per step), wgmma batched (15 DEPBAR / 42 QGMMA), no C75xx | DKSM's traffic stays; the pipelining itself does not help (E1 below) |
| C3 | packed / fp16 accumulators (dK temp or dV window as f16 wgmma accumulators) | packed accumulator | analytic | | | | e4m3 code products reach 448² ≈ 2.0e5 > fp16's 65504, and a 64-term sum more, so f16 accumulation overflows. wgmma has no bf16 accumulator. Packing fp32 to f16x2 after the fact costs instructions on an issue-bound path and frees nothing while the wgmma is in flight |
| E1 | **PIPE probe** (C2 with the dK fold removed: same registers, no LDS / STS), against **base + DKSM + DKH** with the fold removed | second fragment / intra-warpgroup ping-pong | `pipe_nofold` (job 21183029) vs `base_dkh_nofold` (job 21183992); `dkh_nofold_base` (non-DKH) in job 21183029 | timing only | PIPE **−4.05% / −3.51%**; non-pipelined with the same halves **−7.71% / −7.22%**; non-pipelined full-width −8.44% / −7.97% (A spread 3.2% at 32/4) | REG 168, STACK 0-8 | the pipelined order is 3.5-4% slower than the accepted order with identical work and headroom. The tensor / vector overlap it buys is worth less than nothing |
| E2 | **PIPE + RIP + CDP**: dK accumulated in place in the current tile's basis (rescale by g_{m-1} / g_m in C(m), no temp; a named numerics change, dK only), the full-width dK gemm, and the next dP accumulator reused as the carried dS fragment | second fragment with dK in registers | `rip_pipe`, job 21183085 | no (RIP) | **+27.91% / +28.43%** | STACK 72 / 88; main loop 12 STL + 28 LDL per step; no C75xx | about 25 registers over 240 even with CDP. Best case without spills ≈ E1's PIPE + the rescale (≈ PIPE with a fold) → above the accepted time |
| B1 | **two kernels as is**: a dV kernel (S, exp, P codes, dV window) plus a dK + dS' kernel (everything else) | split pass 1 | `ablv` and `ablk` (timing-only switch KDA_S_ABL = 2 / 1), job 21182984 | timing only | dV-only **−43.36% / −42.82%**; dK + dS'-only **−11.20% / −10.61%** → sum **1.455× / 1.466×** | REG 168, STACK 8 | the dV kernel repeats the S gemm and the exp line, which is about 57% of today's pass 1 |
| B2 | **two kernels, each restructured for overlap** (the freed registers used for a pipelined schedule in each) | split pass 1 | bound from B1 + E1 | | ≥ 1.0× even with perfect tensor / vector overlap in both; measured overlap gain ≤ 0 (E1) | | perfect overlap: dV kernel ≥ max(tensor ≈ 0.18, rest ≈ 0.39) and dK kernel ≥ max(≈ 0.28, ≈ 0.61) → ≥ 1.00×; tensor shares from kda-pass1's ablations (gemms −30%, vector −63%). E1 shows the overlap is not even neutral |
| A1 | **2-CTA cluster, dK CTA + dV CTA** on the same key block, P codes over DSMEM (no S recompute) | cluster / DSMEM | bound from B1 | | SM time ≥ 2 × the dK + dS' CTA ≥ **2 × 0.89 = 1.78×** | | each key block occupies two SMs, and the dK side alone keeps 89% of today's step |
| A2 | **2-CTA cluster, balanced**: CTA 0 runs S, exp, P codes and dV; CTA 1 runs dP, the dS line (P fp32 over DSMEM, 32 KB per step), psi, dS' and dK | cluster / DSMEM | bound from B1 (the direct probe `ablks`, KDA_S_ABL = 9, deadlocked; job 21183367 cancelled) | | SM time ≥ 2 × max(0.57, ≈ 0.7) **≈ 1.4×** before the DSMEM traffic | | splitting the work over two SMs pays only if each SM gets more than 2× more efficient. The only mechanism, freed registers for overlap, is measured ≤ 0 (E1) |
| D1 | **coarser psi / k8s contract that removes the per-tile dK fold / temp** (for example psi per 128-row q8s block, or a running psi with RIP that rescales only on basis change) | psi tile / k8s group (named numerics change) | probe `dkh_nofold_base` (no fold at all) | timing only | upper bound **−8.44% / −7.97%** (no fold at all) | | below 10% as an upper bound. A real psi-per-2-tiles needs both tiles' fp32 dS (32 more registers) or a second pass, and the running psi coarsens the dS' codes of off-diagonal tiles (the dQ / dK error rises) |
| D2 | **finer psi (per warpgroup 64 × 64)** | psi tile | p1redesign / S17-psi (cited, `results/screens/psi_log.md`) | | pass 1 −1.2 .. −1.8%; pass 2 +7.4 .. +8.6% | | already measured. It shrinks no accumulator (dK, dV and the temp are per 64 keys either way) |

Supporting compile-only variants (TU jobs 21181828-21182841; `tu/<name>/`):
- `pipe` and `pd1`-`pd4` bisected C7513 to the loop-carried P-code copy.
- `u2` and `u2dkh` (ping-pong buffers, unrolled by two): STACK about 1.9 KB.
- `pc` (PIPE with the full-width temp): STACK 152, 116 STL. This is the register wall itself.
- `rip`, `r1`, `r2`, `d1`: the RIP / CDP orderings. All spill except the DKH ones.

## Per lever (at least two designed variants each)

| lever | variants | best measured / bound | status |
|---|---|---:|---|
| 2-CTA cluster on one key block (DSMEM) | A1 dK / dV CTAs; A2 balanced split with a P exchange | ≥ 1.4× SM time (bound) | failed |
| two kernels (dV, dK + dS') | B1 as is; B2 each pipelined | 1.46× measured; ≥ 1.0× bound | failed |
| dK / dV / temp in smem or packed | C1 DKSM; C2 DKSM + PIPE; C3 f16 / packed (infeasible) | +20.3% | failed (smem bandwidth) |
| psi tile / k8s group contract | D1 coarser (fold removed: bound); D2 finer (per-WG psi) | −8.4% (upper bound) | failed (< 10% as an upper bound) |
| ping-pong / second S / dP fragment | E1 PIPE (DKSM + DKH, fold-free probe); E2 PIPE + RIP + CDP | E1 is 3.5-4% slower than the same work unpipelined | failed |

## Hand-off notes

- Nothing is READY. The switches stay in the source, default off, as a record. DKSM, DKH and PIPE (+ DKSM + DKH) are
  bitwise. RIP is a named numerics change and was never gated. KDA_S_ABL is a timing-only ablation set (wrong
  outputs). ABL & 8 deadlocks; do not use it.
- **The register wall is not pass 1's limiter. The vector instruction stream is.** It runs at about 60% issue
  utilisation with two warps per SMSP, and removing 64 tail instructions (the fold) is worth −8%.
- The levers that can still move pass 1 are fewer instructions on the step's critical path:
  - the exp line, where EXPB is worth about 1.3-1.5% per instruction per element;
  - the dK fold / dV promote tail;
  - the conversions;
  - more issuing warps per SMSP. That would need a design whose per-thread state fits below 160 registers, for
    example a third warpgroup. p1redesign's DVWG bound is −8.8%.
- **Do not retry** with a different register trick:
  - tensor / vector overlap inside a warpgroup was given 64 free registers here and made the step slower;
  - two-kernel and two-SM splits pay a recompute or a 2× SM cost that no overlap can recover.
