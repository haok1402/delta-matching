# S17 forward screen log (D128 causal forward, FlashAttnFwdSm90Fq8F, base d07cdcd = accepted r14_h3)

Each variant has its own worktree, `.local/wt/S17-fwd-NN`, and was screened with `.local/jobs/screen.sh`: a 2-rep A/B against the accepted root at B2 N8192 Hk4, D128 causal matched. 2-rep noise is about +-1%. Negative numbers are faster. Each patch is in `.local/screens/S17-fwd-NN/variant.patch`, and the edit generator is `.local/screens/fwd_scripts/edit.py`.
"bitwise" is the default-route byte-equality check vs the frozen baseline: 72 cases, rc 0 = BITWISE EQUAL. Probes were run with BITWISE=0.
REG/STACK: every forward symbol stays at REG 168. The STACK column gives only the forward symbols that changed against the base set (STACK 32/24/24/24 + 8).

| tag | description | bitwise rc | H16 fwd | H32 fwd | other kernels | REG/STACK | verdict |
|---|---|---:|---:|---:|---|---|---|
| S17-fwd-01 | KSPF v2: the next tile's 4 k8s loads are issued at the end of the previous step (clamped index, so never out of range); the first step is loaded before the loop; D128 only | 0 | -0.67% | -0.55% | within +-0.3% | 168, unchanged | reject (below 2%, matches the v1 ceiling of about 0.5%) |
| S17-fwd-02 | ODEFER: tOrO *= alpha is deferred from the softmax phase into the next PV chain, after the g0 sub-gemm issue (the tail gets it too) | 0 | -0.84% | +3.16% | within +-0.8% | 168, unchanged | reject. SASS: ptxas hoists the 64 FMULs above WARPGROUP.ARRIVE into the barrier section |
| S17-fwd-03 | RHOPRE: rho1..3 and w3 are computed once, under the g0 sub-gemm; the promote uses the cached w3 | 0 | -0.96% | -0.84% | within +-0.4% | 168, unchanged | reject (noise). The base SASS already hoists the rho MUFU/clamps above the barrier |
| S17-fwd-04 | FASKIP: skip tOrO *= alpha when alpha == 1 on every row of the warp (`__all_sync`) | **1** (O, LSE and all downstream differ: e.g. 933/8192 LSE rows on d128_bhnd causal) | +46.90% | +47.38% | ~0 | STACK 168 / 176 (fragment spilled across the branch) | reject: not bitwise in practice (codegen change) and spills |
| S17-fwd-05 | **PROBE** NORHO: no rho rescales in the D128 PV chain (the inner waits stay) | n/a | -8.79% | -9.47% | ~0 | unchanged | probe |
| S17-fwd-06 | NOSCHED: no warp-scheduler barrier at D128 | 0 | +11.58% | +11.01% | ~0 | unchanged | reject |
| S17-fwd-07 | SCHEDQK: the barrier guards only the QK issue (sync after the K wait); the PV chain is unguarded | 0 | +12.06% | +11.40% | ~0 | unchanged | reject |
| S17-fwd-08 | SCHEDPV: the barrier guards only the PV chain (arrive right after the last PV issue, before the promote and QK) | 0 | +1.39% | +1.14% | within +-0.5% | unchanged | reject |
| S17-fwd-09 | VREL: release V right after the PV retire, before the promote | 0 | +0.13% | +0.63% | pass0/1 +1% (noise) | unchanged | reject (neutral) |
| S17-fwd-10 | **PROBE** NODESCALE: no S descale FMULs and no k-scale loads | n/a | -5.02% | -5.26% | pass2 -5% (its outputs change) | unchanged | probe |
| S17-fwd-11 | **PROBE** NOALPHA: no tOrO *= alpha | n/a | -1.07% | -0.67% | ~0 | unchanged | probe |
| S17-fwd-12 | **PROBE** PV1: the PV chain is one 8-atom gemm (no rho, no inner waits), like FA3 | n/a | -3.90% | -4.12% | ~0 | unchanged | probe |
| S17-fwd-13 | ODEFER + PIN: as 02, with warpgroup_fence_operand(tOrO) around the deferred rescale | 0 | -0.42% | -0.63% | ~0 | unchanged | reject. Fences do not pin the code: ptxas still hoists the FMULs above ARRIVE |
| S17-fwd-14 | KWEARLY: the K wait moves before the scheduler-barrier sync | 0 | +10.21% | +10.31% | ~0 | STACK 40 / 48 (new) | reject |
| S17-fwd-15 | **PROBE** NOPROMOTE: per-step promote removed. The compiler then also drops the whole steady-state PV chain (no PV QGMMA left in the loop), so this measures "no PV work at all" | n/a | -16.90% | -18.75% | ~0 | unchanged | probe |
| S17-fwd-16 | QKSPLIT 128/128: QK as two 64x128 halves, with the PV promote issued between them | 0 | +29.29% | +29.43% | ~0 | STACK 168 / 192 | reject (spills) |
| S17-fwd-17 | QKSPLIT 64/192: the first QK part is 64x64 (32 regs) before the promote, the rest 64x192 | 0 | +18.60% | +22.00% | ~0 | STACK 136 / 144 | reject (spills) |
| S17-fwd-18 | QKSPLIT 32/224 | 0 | +11.68% | +12.57% | ~0 | STACK 96 / 104 | reject (spills). SASS: ptxas issues QK part B straight after part A and sinks the promote after both, so tOrD + full tSrS + tOrO are live together and it spills |

No winner: no bitwise variant reached -2% on both shapes, so nothing was rerun.

## What the screens and probes say about forward time (D128, B2 N8192)

- Rho rescales inside the PV chain: about 9% (05). They sit inside the warp-scheduler critical section, between retired sub-gemms, so the tensor pipe idles during them. No bitwise schedule fixed this. An N-split interleave was tried as PVIL in round 7 (-0.08%). A QK/PV overlap needs more than 240 registers.
- S descale: about 5% (10), even though it lies outside the critical section. The softmax phase is on the critical path too. Each 2-warp SMSP is close to its issue and MUFU limits: about 1288 instructions and 149 MUFU per warp per step against about 1400 tensor cycles per WG-step. Bitwise, the descale FMUL cannot be folded, because the exp2 FFMA needs the rounded s*g.
- tOrO *= alpha costs only about 1% (11). So vector work outside the barrier section is cheap, and deferring or skipping it (02, 04, 13) has no upside.
- Removing all steady-state PV work only gains 17-19% (15). Collapsing the chain into one gemm (12) gains just 4%, which is less than removing the rescales and keeping the waits (9%). The tensor pipe is not the limit. The step is bound by the ping-pong of [PV chain + promote + QK issue] against [softmax].
- The scheduler barrier's current placement (PV chain + QK issue) is clearly best: removing it or narrowing it costs +1% to +12% (06, 07, 08, 14).
- Source-order scheduling levers do not survive ptxas. It moves register-only ALU work freely across WARPGROUP.ARRIVE, named barriers and operand fences (02, 13). It also reorders QGMMA issue relative to ALU work (18). So "move X under the wgmma" ideas need a real data dependency or a different register plan, not reordered source.
- setmaxnreg has no headroom: 24 x 128 + 240 x 256 = 64512 of 65536 registers.
- Remaining forward levers are numerics-changing (not bitwise): the rho chain (GRPFOLD-like folds, rho folded into P quantization / v-scale), the S-descale fold into the exp2 constant, and a smaller-tile register plan that would allow FA3's QK(n)/PV(n-1) overlap.
