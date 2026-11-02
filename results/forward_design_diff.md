# Forward kernel: FA3 fp8 forward vs fwd_fp8 (round 7)

Hao's direction after round 6: FA3's fp8 forward is ~25% faster than ours at the same shapes on this machine
(baseline.md, D128 causal N32k 4.51 vs 6.14 ms; D256 causal N8192 0.91 vs 1.51 ms) and ours grew from it. This file
diffs the two designs from source (third_party/flash-attention/hopper at the pinned commit vs csrc/) and from the
emitted SASS of the shipped instantiations (cuobjdump of the FA3 extension and of the accepted DO2 library).

## Configuration

| | FA3 fp8 forward | fwd_fp8 (ours) |
|---|---|---|
| tile (D128) | 128 x 224 (tile_size.h) | 128 x 256 (F3, `FA3FP11_HD128_BN256=1`) |
| tile (D256) | 128 x 128 | 128 x 128 |
| consumer warpgroups | 2 (64 rows each) + producer | 2 (64 rows each) + producer |
| stages | 2 | 2 |
| PV operand | RS (P in registers) | RS |
| V for the PV wgmma | in-smem transpose by the producer | same (copy_Vt_to_V) |
| inter-warpgroup ping-pong | warp-scheduler barrier around the gemm issue | same barrier, around the PV chain + QK issue |
| intra-warpgroup overlap | yes: QK(n) and PV(n-1) issued back to back; softmax(n) runs while PV(n-1) is in flight (`wait<1>` then `wait<0>`) | none: PV-first step, every wgmma group retired (`wait<0>`) before the next vector work |
| PV accumulation | directly into tOrO, one chain of k-atoms per tile, one wait | per-tile zero-initialised temp tOrD; four 64-key (D128) / 32-key (D256) sub-gemms, each retired, the temp rescaled by rho_g between them (Horner), then promoted into tOrO in fp32 (FMA) |
| softmax | one row max, constant Max_offset, exp2 with one FFMA | four group maxima (4 quad all-reduces), per-group clamped basis ms_g, exp2 with one FFMA, group partials folded onto the running-max basis (exp2 per group) |
| scales | per-tensor descale folded into the softmax scale | per-(row block, 64/32-key block) S descale (one FMUL per S element), per-key-block v-scales feeding rho and the promote weight |
| scale loads | none per tile | 4 k-scales (LDG) after the QK retire; 4 v-scales (LDG) at the step end |

Register plan per consumer thread (fp32 regs; the 240-register consumer budget):

| | tOrO | tSrS | PV temp | P codes | peak |
|---|---:|---:|---:|---:|---:|
| FA3 D128 (BN 224), QK(n) and PV(n-1) both live | 64 | 112 | - | 28 | ~204 |
| ours D128 (BN 256), PV-first | 64 | 128 | 64 (tOrD, disjoint from tSrS) | 32 | 64 + max(128, 64 + 32) = 192 + scratch |
| ours D128 if QK(n) were in flight during the PV chain | 64 | 128 | 64 | 32 | 288 (over budget) |
| ours D256 (BN 128, NSPLIT 2), PV-first | 128 | 64 | 64 | 16 | 208 |
| ours D256 with the QK overlap | 128 | 64 | 64 | 16 | 272 (over budget; source comment: spills, outlining, C7510) |

So the FA3 overlap that hides the softmax under the PV wgmma is not available to ours at either head dim without
first shrinking a live fragment; the per-tile temp (the fp8-accumulate precision contract) is what costs it.

## Emitted code (static opcode counts of the whole kernel, D128 full, StaticPersistentTileScheduler instantiation)

| opcode class | FA3 fp8 128x224 | ours 128x256 |
|---|---:|---:|
| total instructions | 2968 | 4216 |
| FMUL | 140 | 828 |
| FFMA | 232 | 414 |
| FMNMX | 230 | 326 |
| MUFU.EX2 | 226 | 290 |
| FADD | 226 | 288 |
| MOV | 115 | 261 |
| QGMMA | 22 | 24 |
| WARPGROUP.DEPBAR | (in "WARPGROUP" 8) | (in "WARPGROUP" 20) |

(static counts include the prologue / first-tile copies; per steady-state step ours adds ~128 FMUL for the S descale
and ~192 FMUL for the three rho rescales of the 64-register temp at D128.)

Steady-state loop order in our SASS (D128 full): `BAR.SYNC` (scheduler) -> four times [`WARPGROUP.ARRIVE`, 2 x
`QGMMA.64x128x32`, `WARPGROUP.DEPBAR.LE gsb0, 0x0`, ~130 instructions of rho / rescale] -> V release -> K wait ->
4 x `QGMMA.64x256x32` -> `BAR.ARV` -> `WARPGROUP.DEPBAR.LE gsb0, 0x0` -> **4 x `LDG.E` (k-scales)** -> descale /
softmax (MUFU) -> 4 x `LDG.E` (v-scales). The k-scale loads are issued only after the QK retire, so their latency is on
every step's serial path ahead of the descale.

## Where the extra time can come from (hypotheses to measure)

1. Loss of intra-warpgroup overlap (PV-first): each warpgroup's step is a serial chain PV chain -> QK -> vector work.
2. PV chain serialization: four retired sub-gemms per tile with rho work between them (~750 cycles of the ~1263-cycle
   PV region in the (perturbing) trace at D128 full, against ~512 cycles of tensor work).
3. Extra vector work: S descale FMULs, grouped max (4 reductions), group fold, promote FMAs, rho rescales.
4. Exposed scale-load latency (k-scales after the QK retire).

Measured by timing-only ablations (results/r7/wi) and the forward trace (results/timeline_fwd.md).
