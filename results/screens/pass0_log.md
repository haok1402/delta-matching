# S17 pass-0 screening log (D128 `DeltaV6<true>`, csrc/fp11_delta_v6.h)

Base: d07cdcd (accepted build at the start of the round). A = .local/frozen/r14_h3. Screen: .local/jobs/screen.sh, 2 reps,
B2 N8192 Hk4, H16 and H32, causal-matched. Numbers are B vs A per kernel. Variants live in .local/wt/S17-p0-NN; patches are in
.local/screens/S17-p0-NN/variant.patch; logs are in .local/logs/kda-S17-p0-NN-<job>.log. Helper scripts are in .local/screens/S17-p0-tools
(misc.py = the edit for every mode, rs.py = RS).

**Note: main moved during this round.** c563139 (P0RTR + P0SPF from kda-sm90-pass0, accepted, pass 0 -5.2%) contains a
quad reduce that is the same as my S17-p0-11 "scatter" (same pairing, 4 SHFL + 4 FADD). It also adds a one-load-per-stage scale
fetch. Everything below is measured against d07cdcd/r14_h3. The parts not yet in main are **nomaskloop**, **skipdead** and
the **setmaxnreg pair 240/24**. They still have to be ported and re-screened on c563139.

## Static profile (r13prod, causal kernel `DeltaV6ILb1E`)

* Structure: 1 CTA/SM (384 thr: producer WG + 2 consumer WGs, each consumer WG owns a 64-row m-block). Smem is about 129 KB
  (Q 16K, dO 16K, K/V 3 stages x 32K). The kernel is compiled at REG:168 (launch bound). The consumer code uses only R0..R161, so the
  setmaxnreg 232 headroom is unused (ptxas register *assignment* still depends on the value, see below). STACK 0.
* Main loop 0x11e0-0x3ed0 = 720 SASS per stage (2 n-blocks), 16 QGMMA 64x64x32 SS, 64 MUFU.EX2. The diagonal-mask path
  (64 ISETP + 64 FSEL + setup, about 146 instructions) is branched around off the diagonal, so the cheapest path is about 574 per stage, about
  287 per n-block per warp.
  - Per element: 6 instructions that bitwise identity forces (FFMA s*dsc-lse, FADD +8, FMNMX 12, MUFU.EX2, FMUL dP*dp, FFMA
    rowsum). That is 192 per n-block.
  - About 95 per n-block are overhead: about 32 uniform-datapath descriptor ops (UIADD3/UMOV 0x80000020/ULEA), 28 Allreduce +
    fold (8 SHFL, 8 FADD, 4 FMUL 2^-8, 4 F2F, 4 DADD), scale prefetch (LDG + UIMAD.WIDE + MOVs, about 12), mask test /
    branch (about 6), QGMMA/WARPGROUP (12).
* Waits: wait<2> (no-op), wait<3> (S1), wait<2> (dP1), and wait<0> at the back-edge. The back-edge wait<0> is needed: removing it
  makes ptxas serialize, C7514 (+33%).
* Resource balance per SM per tile pair: tensor 512 clk (= floor), MUFU 64 warp-instr x 8 clk = 512 clk per SMSP (same as
  the floor), issue about 2 x 287 = 574 slots per SMSP. With only 2 consumer warps per SMSP (registers allow 1 CTA), the kernel
  is **issue-bound**. SS smem traffic (128 B/clk at peak) is not the limiter: RS made no difference.

## Probes (timing only, BITWISE=0): where the time goes

| tag | probe | H16 pass0 | H32 pass0 | reading |
|---|---|---:|---:|---|
| S17-p0-07 | exp2 removed (no MUFU, -1 instr/elem) | -11.79% | -12.79% | MUFU is a minor share |
| S17-p0-08 | dP GEMM removed (C7514 serialized) | -3.35% | -4.39% | half the tensor work, worth only about 4% |
| S17-p0-09 | both GEMMs removed (elementwise + TMA only) | -23.99% | -24.70% | elementwise alone = **76%** of base (about 48 ms on the set, > 39.3 floor) |
| S17-p0-10 | exp + rowsum removed (GEMM + pipeline only) | -40.36% | -40.10% | GEMM path alone = 60% of base, about 38 ms: **at the tensor floor** |
| S17-p0-15 | scatter + "+8" folded into the FFMA addend (numerics change) | -7.44% | -8.28% | -1 instr/elem is worth about 4% more |
| S17-p0-16 | scatter + dP descale per 8-key partial (P0V7_LAZY-style) | -5.71% | -5.54% | -1 FMUL/elem (+2) is worth about 2% more |

Conclusion: time tracks instruction count about 1:1 (scatter cuts 3% of the loop instructions and gives about 3.6%). The GEMM path is already at
the floor. The elementwise stream alone exceeds the floor (about 122%). 90% of the floor would need the elementwise stream below about 0.9x the floor and
perfectly overlapped. Under the bitwise contract the 6 instructions per element are fixed, so 192/n-block is the hard minimum
(issue alone is about 84% of the floor). The realistic bitwise ceiling looks like about 70-75% of floor. Reaching 90% needs a numerics arm (fold8 + lazy dP + lazy fold).

## Variants (bitwise rc = default-route byte equivalence, 72 cases)

| tag | description | bitwise rc | H16 pass0 | H32 pass0 | other kernels | REG/STACK (maxR) | verdict |
|---|---|---|---:|---:|---|---|---|
| S17-p0-01 | UNROLL3 (ring slot compile-time, 3x unrolled) | 0 | +3028.95% | +3056.18% | noise | 168/1040 (228), C7510/C7520 | reject (step outlined into a call, wgmma serialized) |
| S17-p0-01b | 01 + force-inline lambdas | 0 | +1.47% | +1.40% | noise | 168/0 (201) | reject |
| S17-p0-02 | K/V stages 3 -> 4 | 0 | -1.42% | -0.30% | noise | 168/0 (162) | reject (noise) |
| S17-p0-04 | K/V stages 3 -> 2 | 0 | +3.07% | +1.38% | noise | 168/0 (162) | reject |
| S17-p0-03 | RS wgmma: Q/dO held in registers (A from RF) | 0 | +0.44% | -0.12% | noise | 168/0 (192) | reject (smem BW not the limiter) |
| S17-p0-05 | drop the back-edge wait<0> | 0 | +33.47% | +33.89% | noise | 168/0, C7514 | reject |
| S17-p0-06 | setmaxnreg 240/24 (was 232/40) | 0 | -1.71% | -1.79% | noise | 168/0 (165) | marginal |
| S17-p0-06r | rerun of 06 | 0 | -1.78% | -2.04% | noise | 168/0 (165) | confirmed about -1.9% (register assignment effect) |
| S17-p0-11 | quad reduce-scatter (4 SHFL + 4 FADD + 1 fold row, same as main's P0RTR) | 0 | -3.61% | -3.52% | noise | 168/0 (161) | win (already in main as P0RTR) |
| S17-p0-11r | rerun of 11 | 0 | -3.83% | -3.52% | noise | 168/0 (161) | confirmed |
| S17-p0-12 | grid swap (bh fastest, longest m-pair first across all heads) | 0 | -3.75% | +0.20% | noise | 168/0 (162) | inconclusive (shape-dependent tail) |
| S17-p0-13 | 11 + 240/24 | 0 | -3.68% | -3.75% | noise | 168/0 (161) | = 11 |
| S17-p0-14 | 11 + 240/24 + 4 stages | 0 | -3.44% | -3.38% | noise | 168/0 (161) | = 11 |
| S17-p0-17 | 11 + nomaskloop (stages below the diagonal use a step without the mask test: 532-instr loop) | 0 | -0.76% | +0.13% | noise | 168/0 (161) | reject as built (232/40) |
| S17-p0-17r | rerun of 17 | 0 | -0.04% | -0.18% | noise | 168/0 (161) | confirmed about 0 |
| S17-p0-18 | 11 + nomaskloop + 240/24 | 0 | -5.42% | -5.74% | noise | 168/0 (161) | **win** |
| S17-p0-18r | rerun of 18 | 0 | -5.49% | -5.95% | noise | 168/0 (161) | **confirmed** |
| S17-p0-19 | 11 + skipdead (last stage: skip exp/rowsum of the n-block wholly above the diagonal, contributes exactly +0) | 0 | -3.02% | -2.53% | noise | 168/0 (159) | about 11 (small, N-dependent) |
| S17-p0-20 | 11 + nomaskloop + 240/24 + skipdead | 0 | -5.94% | -6.17% | noise | 168/0 (161) | **best** |
| S17-p0-20r | rerun of 20 | 0 | -5.97% | -5.56% | noise | 168/0 (161) | **confirmed** |
| S17-p0-21 | nomaskloop + 240/24 (no scatter) | 0 | +4.56% | +4.84% | noise | 168/0 (163) | reject: shows the register-assignment lottery |
| S17-p0-22 | 11 + nomaskloop + 224/56 | 0 | -2.00% | -2.05% | noise | 168/0 (161) | worse than 240/24 |
| S17-p0-23 | 11 + nomaskloop + 216/72 | 0 | -4.74% | -3.91% | noise | 168/0 (159) | worse than 240/24 |

("other kernels" = fwd / pass1 / pass2 / glue, all within about +-1% in every row. These kernels are unchanged, so their deltas are run noise.)

### Register-assignment sensitivity (important for anyone porting these)
17 and 18 differ only in the setmaxnreg constants, yet they measure -0.0% vs -5.6% (each twice, rep spread under 1%). The main loops
have the same instruction mix (532 / 702 SASS). Neither kernel goes above R161, so the cap is never binding. The SASS diff is 2078 lines of
**register renaming**, the same pattern behind 21 (+4.6%) vs 18. For an issue-bound loop this points to register-bank / operand-
reuse conflicts that depend on ptxas's assignment (an inference; not measured with ncu). Any port of nomaskloop onto c563139
must re-sweep the setmaxnreg pair (232/40, 240/24, 216/72), and ideally confirm with ncu (stall_*: register bank / dispatch).

## Best variant
S17-p0-20 (patch .local/screens/S17-p0-20/variant.patch): bitwise rc 0, pass0 -5.94% / -6.17%, no spill or serialization diagnostics.
S17-p0-18 (the same without skipdead) is confirmed twice at -5.4..-6.0%. Relative to main's c563139 (which already has the scatter), the new
content is nomaskloop + skipdead + 240/24. Expected additional gain is about 2-2.5%, provided the register assignment cooperates.
Next steps: port onto c563139 and re-screen with an A root of c563139. Hao should decide whether to open a numerics arm (fold8 + lazy dP: an additional
about -6% measured as probes, plus a lazy fold), because the bitwise per-element instruction floor caps pass 0 well below 90% of the tensor floor.
