# Round 20, worker S20-p1: pass 1 (FlashAttnBwdSm90Fp8R, D128) layout / parameter space on r19_fwd

Base: kda-sm90 c8c2f48. A = `.local/frozen/r19_fwd`.

Screen = `.local/jobs/screen.sh`:
- 2-rep smoke A/B, B2 N8192, H16/4 and H32/4, causal matched.
- Noise is about +-1%. In this round the other, untouched kernels moved by up to +2% in some jobs, e.g. 01 at H32.
- Bitwise = `equivalence.py compare` on the default route (72 cases) against `.local/equiv4/r17pk_default.pt`.

Locations:
- Trees: `.local/wt/S20-p1-NN`.
- Libraries: `.local/jit/S20-p1-NN`.
- Patches: `.local/screens/S20-p1-NN/variant.patch`, a diff against the pristine HEAD archive `.local/rounds/r20-p1/pristine`. Copies are in `.local/rounds/r20-p1/patches/`.
- Tools: `.local/rounds/r20-p1/tools/{mktree,setflag,mkpatch}.sh`.
- Release flags were changed consistently in `delta_matching/_buildcfg.py` FLAGS and `csrc/dm_flagset.h`.

REG/STACK: every variant shows the same Fp8R resource line as r19_fwd (REG 168 at launch, i.e. 240 after setmaxnreg; STACK 0). No ptxas spill or C75xx diagnostics.

## Measured variants (all bitwise EQUAL on the 72 default-route cases)

| tag | layout / parameter | smem | pass1 H16 | pass1 H32 | verdict |
|---|---|---:|---:|---:|---|
| 01 | DOST 1 -> 2 (Q / dO share one 2-stage pipeline) | +8 KB | +0.46% | +2.59% | reject (no longer helps on r19_fwd; the r17-era -2% gain is gone) |
| 02 | DOTST 1 -> 2 | +8 KB | +1.23% | +2.18% | reject |
| 03 | QST 3 + DOST 2 | +24 KB | +0.34% | -0.05% | neutral |
| 04 | DOST 2 + DOTST 2 | +16 KB | -0.14% | -0.46% | neutral |
| 05 | PERSIST: dynamic persistent scheduler (grid = 132), same L2-sectioned LPT tile order as FA3's SingleTileBwdLPTScheduler, global atomic tile counter (self-resetting), KVEmpty named barrier (the producer loads the next tile's K / V once the consumers retire the last S / dP), epilogue bf16 staging moved onto the dV overlay (so the next tile's Q / K / V loads can land during the epilogue), one 256-thread sync after the epilogue | = | +1.23% | +1.69% | reject |
| 05c | rerun of 05 (same library) | | +2.07% | +1.16% | confirms: slower |
| 06 | 05 without the post-epilogue sync and without the per-tile overlay re-zero (the first promote of a tile writes `__fmaf_rn(v, f, 0)`, bitwise the accumulate onto +0) | = | +0.58% | +1.88% | reject |
| 07 | LPT L2 section = 1 (b, h) instead of 4 (non-persistent) | = | +4.49% | +0.82% | reject |
| 08 | LPT L2 section = 16 (b, h) | = | +0.45% | +0.32% | neutral |

Jobs:

| variant | job | log |
|---|---|---|
| 01 | 21198733 | `.local/logs/kda-S20-p1-01-21198733.log` |
| 02 | 21198736 | `.local/logs/kda-S20-p1-02-21198736.log` |
| 03 | 21198735 | `.local/logs/kda-S20-p1-03-21198735.log` |
| 04 | 21198734 | `.local/logs/kda-S20-p1-04-21198734.log` |
| 05 | 21199055 | `.local/logs/kda-S20-p1-05-21199055.log` |
| 05c | 21199439 | `.local/logs/kda-S20-p1-05c-21199439.log` |
| 06 | 21199437 | `.local/logs/kda-S20-p1-06-21199437.log` |
| 07 | 21199534 | `.local/logs/kda-S20-p1-07-21199534.log` |
| 08 | 21199533 | `.local/logs/kda-S20-p1-08-21199533.log` |

The persistent build (H7 in plan.md, never built before) is correct: bitwise, no deadlock. It cannot win much, and here it loses:
- The non-loop part of a CTA is small, by the round-2 profile:
  - prologue about 2000 cycles;
  - epilogue about 780;
  - exit about 590;
  - a mean tile of about 65 steps x about 3300 cycles.
- That bounds the recoverable overhead at about 1.5-2%.
- The persistent loop adds per-tile costs: scheduler handoff barriers, the KVEmpty rendezvous, and live scheduler state. These outweigh the overlap.

## Not built: knobs that are infeasible or already measured by others

| knob | reason | evidence |
|---|---|---|
| kBlockM 128 (two psi tiles per step) | S^T + dP^T for 64 keys x 128 rows = 128 fp32 registers, plus dK 64 + dV 64 = 256 > 240 before the dK temp. Splitting rows across WGs needs dK / dV for 128 keys per WG (256 registers). | kda-p1redesign (register arithmetic); SPLITN / SPLITS (32-row sub-steps) measured +8.8 / +3.6% |
| kBlockN 64 (1 consumer WG per CTA) | A 64x128 psi tile would straddle 2 CTAs: a cluster amax exchange per step, or per-64x64 psi (HALFPSI: pass 2 +7..13%). The decoupling it buys is measured at -1.7% (FR16 WGPSI) to -2.9% (HALFPSI, pass 1 only). | kda-p1redesign, kda-p1fresh, kda-pass1 |
| 2 CTAs per SM, smaller tiles | Registers: 2 x (256 x 240 + 128 x 24) = 129k > 65536. Even a 1-WG CTA needs 2 x (128 x 232 + 128 x 24) = 65536, exactly the file with no slack. Smem: one CTA uses 158.7 KB, of which the dV overlay alone is 64 KB. | arithmetic |
| 3 consumer WGs (192 keys or a dV WG) | At 3 WGs a consumer WG gets <= 160 registers, but dK 64 + dV 64 + S / dP 64 = 192 is needed. The DVWG bound is -8.8% before costs. The real 4-WG role split is +8..10%; the dV-only gemm WG is +21%. | kda-p1redesign (ABLDV), kda-p1fresh (FR66 / FR67) |
| register split | The consumers are already at the ceiling: 256 x 240 + 128 x 24 = 64512 of 65536. | arithmetic |
| dS' store layout | DSNAT (dS' in the dK A-fragment layout, transpose moved into pass 2): pass 1 -0.1 / +0.1%, pass 2 +1.9%. The transpose is hidden under the dK gemm. A key permutation that removes the 8 SHFL + 8 PRMT (with a matching Kt twin order) would remove less than DSNAT did. | kda-p1redesign |
| dQ workspace | Pass 1 writes no dQ. FUSEDQ's bytes floor (290 ms) is above pass 1 + pass 2 today (225 ms). | kda-p1redesign |
| dV overlay layout | float4 overlay (PROMOTE-V4, DVVEC): neutral and STACK 0 -> 8. | S17-p1-25, S19-p1-03 / 06 |
| 128-row Q / dO stages (BM128-style pairing) | Saves about one wait + release per pipeline every two steps, about 1% of the per-step instructions at most. The removable bookkeeping sits in gemm-wait shadows (S19-p1 reading), so it was not built. | analytic |
| cluster / TMA multicast of Q / dO | L2 reads are about 2.3-3 TB/s, below the limit, and the pipeline already hides load latency (QST / DOST sweeps above are neutral). It needs co-scheduled CTAs in lockstep (causal ranges differ by 2 m-blocks). FLASHATTENTION_DISABLE_CLUSTER is a release pin. | analytic |

## Verdict

**No candidate.** Every layout / parameter variant inside the current kernel is bitwise but neutral or slower:
- pipeline depths: best -0.14 / -0.46%;
- persistent scheduling: +0.6 .. +2.1%;
- LPT section size: best +0.3%.

The remaining layout knobs are register-infeasible on sm90 (128-row steps, 2 CTAs/SM, 3 consumer WGs, a larger register split). Or they were already measured neutral or slower by kda-p1redesign / kda-p1fresh / kda-pass1:
- 64-key CTAs;
- dS' native layout;
- dV overlay layout;
- fused dQ.

The current geometry is the only feasible point of the space under the psi-per-64x128 / k8s-per-64-keys contract and the 240-register budget:
- 64 rows x 128 keys;
- 2 consumer WGs;
- 240 / 24 registers;
- QST 2 / DOST 1 / DOTST 1;
- non-persistent LPT with 4-head sections.
