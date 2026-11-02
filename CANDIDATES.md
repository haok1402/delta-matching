# kda-p1redesign candidates (pass 1 structural redesigns, D128 causal matched)

Helper kda-p1redesign, branch `kda-sm90-p1redesign` (on `origin/kda-sm90` 3106faf = r17_p1 accepted). Every design is a
compile-time switch in `csrc/mainloop_bwd_sm90_fp8r.hpp` (and `fq8_dq_pass2_hd128.h` for DSNAT), default OFF: with no
switch set the build is the accepted source (C7517 notes are pre-existing: the no-switch `base` build has the same two).
Built with `DM_XDEFS` / `DM_XFLAGS` (`_buildcfg.py` hook, same as kda-pass1). Scripts and raw logs:
`kda-work/p1r-aux/jobs/{build,screen,eq}.sh`, `jobs/logs/`, `runs/`, `builds/<name>/{resource.txt,build.log}`.

Rule for redesigns (brief): build and time a minimal prototype first; drop a design whose prototype cannot beat the
accepted pass 1 (frozen `r17_p1`) by 10%. Screens: `ab.py --grid one --smoke`, 2 alternating reps, B2 N8192 at Hq/Hk
32/4 and 16/4, A = frozen r17_p1, same job. Numbers: pass 1 median change B vs A (32/4, 16/4); spreads <= 0.9%.

**Result: no READY candidate.** No structural redesign reaches the 10% bar; the best is a timing-only upper bound of
-8.8% (DVWG), which a real implementation cannot reach. Every other prototype is neutral or slower. Details and the
reasons follow, then a summary per lever.

## What bounds the accepted step (structure, from kda-pass1's ablations and these prototypes)

- Tensor and vector work of a warpgroup do not overlap (kda-pass1: all gemms removed -30%, all vector work removed -63%,
  additive). Overlap needs a second S / dP fragment in flight while the vector work runs. The register file does not
  allow it: dK 64 + dV 64 + dK temp 64 + S / dP 64 fp32 per thread are already >= 240 at the peak.
- The 64 x 128 psi tile and the 64-key k8s groups fix the CTA at 128 keys and a warpgroup at 64 keys, so the three
  64-register accumulators per warpgroup cannot shrink without a contract or numerics change.
- Measured here, ptxas adds two hard constraints for cross-step overlap: (a) a wgmma accumulator consumed in the NEXT
  loop iteration serializes every wgmma of the function (C7511, even when it is issued and waited at the very end of a
  step: diagnostic build `early9`); (b) an accumulator whose physical registers were written by a non-wgmma
  instruction inside a still-open wgmma stage serializes the function (C7515). Only issue points AFTER the dV gemm has
  retired avoid both. At those points there is no vector work left to hide the next S under.

## REJECTED (prototype measured; numbers are pass 1 change B vs A, 32/4 and 16/4)

| design | lever | prototype / build | pass 1 | other | reason |
|---|---|---|---:|---|---|
| **DSNAT**: dS' codes stored in the dK A-fragment's native layout (2 STS.128 + one 4-KB bulk copy per warpgroup half; no shuffles / prmt / stmatrix.trans); the transpose moves into pass 2 (ldmatrix.trans + prmt into an RS A operand) | dS' handling, wider instructions | `dsnat`, job 21177075; SASS: SHFL 51 -> 19, PRMT 40 -> 8, STSM.trans gone | -0.09% / +0.10% | pass 2 +1.88% / +1.82% | the transpose is already hidden under the dK gemm (H1b): removing it gains nothing. dK / dV / psi bitwise (job 21177076); my pass-2 row map was still wrong (dQ differs), not debugged since the design is dead |
| **DVWG**: a third consumer warpgroup owns dV for all 128 keys (P codes handed over in smem), so dV rescale / gemm / promote leave the two compute warpgroups. Bound: ABLDV = the compute warpgroups with all dV work deleted and the P codes stored to smem (dV garbage, timing only) | consumer warpgroups (3), FA3-style warp specialization | `abldv`, job 21177075 | **-8.77% / -8.44%** (upper bound) | with DSNAT: -8.31% / -7.59% | the bound is below 10% before the third warpgroup's issue slots, hand-off barrier and the 176 / 160 register split are paid (the split also leaves the compute warpgroups no room for any overlap) |
| **SPLITN**: S^T and dP^T as two m64n32 halves (query rows 0-31 / 32-63) in separate wgmma groups; exp / dS of the first half run while the second half is on the tensor pipe | tile shape (32-row sub-steps) | `splitn`, job 21177393; bitwise on the default route (72 cases, job 21177395) | **+8.82% / +8.95%** | | n32 SS atoms re-read the K / V A operand from smem per half and double the QGMMA count; ptxas also hoisted the second-half wait next to the first |
| **SPLITS**: only S^T split in halves (dP one gemm) | tile shape, variant 2 | `splits`, job 21177603 | **+3.57% / +3.97%** | | same cost, half the size; the exp overlap does not pay for it |
| **RIP**: dK accumulated in place in the current tile's basis (rescale by g_{m-1} / g_m before the dK gemm; no 64-register temp, no fold). Named numerics change (dK only) | registers for overlap | `rip`, job 21177981; REG 168, STACK 0, no C75xx | **+3.78% / +3.52%** | | the rescale (a division + 64 FMUL) sits between the psi barrier and the dK issue, where the fold sat after the dK retire |
| **EARLY3 / EARLY4**: RIP + the next S^T issued right after the dV gemm into P's dead registers (+ next dP^T after the dK gemm) | cross-step overlap | `early3`, `early4`, job 21177727 | not timed | | C7511 (wgmma serialized), STACK 256: ptxas cannot keep the next S as an accumulator across the back-edge |
| **EARLY5 / EARLY7** (+ one loop with a runtime mask predicate): next S^T issued after the dK gemm (5) or after the dV retire (7), everything retired at step end | cross-step overlap | `early5`, `early7`, job 21178151; `early5l`, `early7l` | **+33.6% / +35.2%** (5), **+32.8% / +33.9%** (7) | | still C7511: every QGMMA serialized |
| **EARLY5c / EARLY7c**: as above with the S result copied from a step-local accumulator into a plain carry fragment | cross-step overlap | `early5c`, `early7c`, job 21178441 | **+24.4% / +24.3%** (5c), **+22.8% / +23.0%** (7c) | | C7515: ptxas reuses registers written inside the open dV / dK stage for the next S; still fully serialized |
| **EARLY7s**: EARLY7c with the next S accumulated into the dS fp32 registers (last written before the dK arrive) | cross-step overlap | `early7s`, job 21178574; no C75xx, wgmmas batched again | **+9.95% / +11.12%** | | legal only after dV retires: the next S then queues behind dK with no vector work left to hide it, and RIP's +3.6% remains. EARLY6c / 8c (next dP too): STACK 120-200 + C7515 |
| **FUSEDQ**: dQ accumulated in pass 1 (pass 2 and its dS' round trip dropped) | dQ handling, dS' on chip | analytic (sol4-v2, `results/r17/accepted_c3/sol4v2.md`) | | | dQ partials are fp32: >= 32 KB read + 32 KB written per 64 x 128 tile vs today's 8 KB dS' write (+ 8 KB pass-2 read). Pass 1's bytes floor is 36.6 ms at 8 KB / tile over the four shapes, so a fused kernel's bytes floor is >= 8 x 36.6 = 290 ms. That is above pass 1 + pass 2 today (187.8 + 37.6 = 225 ms) before determinism (no atomics: n-ordered semaphores) is paid. Keeping dS' on chip needs the same fusion |
| **128 query rows per step** (two psi tiles per step) | tile shape | register arithmetic | | | S^T + dP^T for 64 keys x 128 rows = 128 fp32 registers per thread, plus dK 64 + dV 64 = 256 > 240 before the dK temp; with the rows split across the warpgroups instead, each warpgroup needs dK / dV for all 128 keys (256 registers) |
| **64 keys per CTA** (1 consumer warpgroup, 2 CTAs / SM at 232 + 24 registers, psi max across a 2-CTA cluster) | tile shape / CTA keys | proxy measurements | | | its only gain is decoupling the two 64-key halves. Measured decoupling on this kernel: per-warpgroup psi without the barrier -2.1 .. -2.9% (kda-pass1 HALFPSI, which also needed a +7..13% pass 2). A one-time warpgroup offset (PP1) gave -1.1 .. +1.0%: a lone warp per SMSP issues the vector work no faster. The cluster amax exchange per step re-adds the coupling |

## Per lever (brief: at least two designed variants each)

| lever | variants | best | status |
|---|---|---:|---|
| tile shape: 64 vs 128 query rows per step | SPLITN (32-row halves of S and dP), SPLITS (S only); 128 rows: register arithmetic | +3.6% | failed |
| tile shape: 128 vs 64 keys per CTA | 64-key CTAs with a cluster psi (proxy: HALFPSI / PP1 measurements); 192-key CTAs with 3 warpgroups (psi tile 64 x 128 does not divide; registers) | ~-2..-3% (proxy) | failed |
| consumer warpgroups (2, 3) | DVWG bound (ABLDV), DVWG + native dS' (dsnat_abldv); dK warpgroup infeasible (dK 128 + temp 128 registers) | -8.8% (bound) | failed (< 10% as an upper bound) |
| FA3-style warp specialization and dQ handling | DVWG (dV warpgroup); FUSEDQ (in-kernel dQ, analytic) | -8.8% (bound) | failed |
| dS' on chip / fused dQ | FUSEDQ (bytes floor 290 ms > 225 ms); DSNAT (transpose moved to pass 2) | 0% | failed |
| fewer, wider instructions | DSNAT (2 STS.128 for 16 SHFL + 16 PRMT + 2 STSM); LSE / dPsum already LDS.64 in the accepted SASS; MUFU / F2FP / FFMA have no wider bitwise form on sm90 | 0% | failed |
| cross-step overlap via freed registers (RIP + EARLY) | EARLY 3/4/5/6/7/8 (+ copy, + one loop, + S into dS registers) | +10.0% (EARLY7s) | failed (ptxas C7511 / C7515 at every issue point that would overlap) |

## Hand-off notes

- Nothing here is READY for kda-sm90. The switches stay in the source, default off and compiled out, as a record. The
  no-switch build is the accepted kernel. DSNAT's pass-2 half is unfinished (dQ row / key map); do not enable it.
- For anyone continuing the overlap direction: a wgmma accumulator must be produced and consumed inside one iteration,
  and its physical registers must not be written inside another open wgmma stage. On the accepted schedule that leaves
  only "after dV retires" as an issue point for the next S, which hides nothing. A real overlap needs 64 more free
  registers per thread (e.g. both dK and dV out of the compute warpgroups), which the 128-key psi tile does not allow.
