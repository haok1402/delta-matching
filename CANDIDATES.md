# Pass 0 (matched-delta prepass, D128) candidates — branch kda-sm90-pass0 (helper kda-pass0)

Target: `csrc/fp11_delta_v6.h` (DeltaV6, the row-CTA pass 0: 128 query rows per CTA = 2 consumer warpgroups x 64 rows,
K / V streamed in 128-key stages through a 3-deep TMA ring, per 64x64 tile S = Q K^T and dP = dO V^T as 4 + 4
`QGMMA.64x64x32` SS, exp line + fixed-order row sums + fp64 fold in registers). Acceptance shapes: D128 causal matched,
B2 N8192 and B1 N65536 at H/Hk 16/4 and 32/4 (`kernels.PAPER_SET`), geometric mean of per-shape pass-0 ratios.
A = frozen accepted `r14_h3` (`.local/frozen/r14_h3`, library 756158cb1226).

Tooling (outside the repo, `~/workspace/haok1402/kda-work/p0/`): `tools/mkvar.sh` rebuilds only the pass-0 TU from a
patched copy of HEAD's csrc and relinks it with the accepted r13prod objects (the rebuilt unpatched TU is SASS-identical
to the frozen library's pass 0, md5 of both instantiations); `tools/p0t.py` times fwd+bwd per kernel (CUPTI, the default
route) on the four shapes, roots alternated per repetition in fresh processes; `tools/screen.sh` builds + equivalence
(default route, 72 cases, vs the frozen dump) + timing in one short job; `tools/abjob.sh` runs `benchmarks/ab.py run
--grid paper --target pass0` (3 alternating reps). Patches: `p0/tools/*.patch`.

## Profile (round P0-1; jobs 21171021, 21171179; no ncu)

Per-shape pass 0 of the accepted build: 0.659 / 19.9 / 1.288 / 40.1 ms (B2 H16 8k / B1 H16 64k / B2 H32 8k / B1 H32 64k);
about 16% of matched fwd+bwd kernel time (total 4.04 / 117 / 7.87 / 234 ms).

Timing-only what-if builds (pass 0, B2 H16 8k; all four shapes move together):

| what-if | pass 0 ms | change | reading |
|---|---:|---:|---|
| base (rebuilt, = frozen SASS) | 0.659 | 0 | |
| no vector work (no mask / exp / row sum) | 0.399 | -39% | tensor + loads alone = the calibrated tensor floor (~0.403 ms by the sol4 count) |
| no wgmma (vector work on zero accumulators) | 0.501 | -24% | vector work alone |
| no dP gemm (S only) | 0.641 | -2.7% | halving tensor work barely matters |
| no MUFU (exp argument kept) | 0.583 | -11.5% | |
| no causal mask | 0.661 | 0 | mask only on diagonal tiles already |
| all CTAs load n-block 0 (L2-hot) | 0.867 | +31% | hot-spotting, not informative |
| V not loaded (half the K/V bytes) | 0.649 | -1.7% | L2 / HBM bandwidth is NOT the limiter |
| no vector + V not loaded | 0.396 | -0.8% vs no-vector | |
| TMA ring depth 2 / 4 (flag) | 0.681 / 0.654 | +3.3% / -0.8% | |

Static SASS (causal instantiation, steady-state stage loop = 2 n-blocks per warp, mask blocks excluded): 564 instructions
per stage: 384 per-element (FFMA 128, FMUL 64, FADD 64, FMNMX 64, MUFU 64: six per element, fixed by the matched
identity with pass 1's exp line and the bitwise row sum), 56 reduction (8 SHFL + 8 FADD per tile, 2^-8, fp64 fold),
16 QGMMA + 8 WARPGROUP + 3 SYNCS, ~95 integer / uniform (wgmma descriptors ~45, scale-prefetch addressing ~20, pipeline
state). Per SM a stage costs ~2590 clk measured vs 1128 issue clk per scheduler (2 warps x 564): pass 0 is not
issue-bound (44% issue use); the vector work alone runs at 57% issue use (2 warps per scheduler). Tensor+load-only
equals the tensor floor, vector-only is 1.24x it, the full kernel is 1.64x it: the gap is overlap / latency, not
throughput. Instruction cuts still pay roughly 1:1 (RTR: -3.5% instructions, -3.7% time).

Duplication with the forward / pass 1: pass 0 recomputes S and dP (half of pass 1's four gemms and all of its exp
work) because delta must be complete for every row before pass 1 forms dS = P (dP - delta) and its per-tile psi; pass 1
is column-CTA (one 128-key column, all rows) so a row's delta is only complete after every column CTA. Reusing pass 0's
P / dP in pass 1 would need them stored (2 x B H N^2 / 2 bytes in fp8 causal: 2 GiB at B2 H16 8k) and would change the
numerics (fp8-rounded P / dP feed dS); fusing pass 0 into the forward is impossible (dO is not known). Pass 0 stays a
separate row pass; levers are inside it.

## Candidates

| # | name | commit / patch | change | bitwise | screen (geomean pass 0) | formal A/B (2 jobs) | REG / STACK / LOCAL | status |
|---|---|---|---|---|---|---|---|---|
| C1 | RTR | 9635adb (`rtr.patch`) | transposed quad reduce: 4 SHFL + 4 FADD instead of 8 + 8, one fp64 fold per lane | yes (default route, 72 cases) | -3.74% (job 21171179, 3 reps) | ab.py paper / pass0: **-3.88%** (job 21171446; per shape -3.83 / -3.62 / -4.40 / -3.68) and **-3.66%** (job 21171447; -3.83 / -3.85 / -3.40 / -3.57); ACCEPT in both, no shape slower beyond spread | 168 / 0 / 0, no spills, no ptxas diagnostics | **READY** |
| C2 | RTR + SPF | 41bfa7b (on C1; `spf.patch`) | stage scales: one float2 load per tensor per stage at s + 1, no clamps (531 vs 544 instr / stage) | yes (72 cases) | -5.46% (job 21171431; -5.79 / -4.99 / -5.79 / -5.27) | ab.py: **-5.27%** (job 21171806; -4.99 / -5.29 / -5.33 / -5.46) and **-4.97%** (job 21171794; -4.61 / -4.87 / -5.03 / -5.36), ACCEPT in both | 168 / 0 / 0, no spills | **READY** (supersedes C1) |
| C3 | + LPT + W4 (on C2) | c9f9cda (LPT), 1b48691 (W4), aef8c7a (template warpgroup count, 128-row fallback when seqlen_q % 256 != 0) | grid (bh, p) for global longest-first dispatch; 4 consumer warpgroups per 256-row CTA (112 regs, one S/dP pair per warpgroup, past-diagonal n-blocks skipped) | yes: full equivalence vs frozen r14_h3 on default / kv / head / streams2 (72 cases each, job 21172311); aef8c7a leaves both W4 kernels SASS-identical to the timed build and its forced 2-warpgroup path is bitwise on 72 cases (job c5) | -14.03% stack incl. split (21171938) | ab.py: **-12.40%** (job 21172311; -15.63 / -9.80 / -14.91 / -9.05) and **-12.32%** (job 21172459; -16.02 / -9.59 / -14.08 / -9.40), ACCEPT in both | 96 launch / 112 consumer / 24 producer, STACK 0, LOCAL 0, no spills, no ptxas diagnostics | **READY** (stack on C2) |
| C4 | + P0ALL (on C3) | de91150 (`delta_matching/split.py` only) | split route at D128 matched: pass 0 once on all heads (one launch), each group takes its slice | outputs: O / dQ / dK / dV (and the captured deltas) byte-equal to C3 on default / kv / head / streams2, 72 cases each (`p0/tools/eqout.py`, job 21172334). `benchmarks/equivalence.py dump` refuses the split routes by its schema (it expects one delta_fp8 capture per group), so its expected-schema derivation needs one all-heads delta call on matched D128 split routes | -14.03% (21171938) | ab.py (C2+C3+C4 vs r14_h3): **-14.35%** (job 21172334; -16.05 / -13.72 / -14.91 / -12.70) and **-14.47%** (job 21172595; -16.03 / -13.56 / -15.28 / -12.95), ACCEPT in both | Python only | **READY** (stack on C3; needs the equivalence schema update) |

## Screen log (every variant; 2-rep screens unless noted; geomean of per-shape pass-0 ratios vs frozen r14_h3)

| variant | patches / flags | bitwise (72) | geomean | per shape (8k H16, 64k H16, 8k H32, 64k H32) | job | note |
|---|---|---|---:|---|---|---|
| rs | Q / dO in registers (RS wgmma) | yes | -1.65% | -1.45 / -1.66 / -1.82 / -1.70 | 21171179 (3 reps) | |
| rtr | quad reduce | yes | -3.74% | -3.84 / -3.54 / -4.02 / -3.58 | 21171179 (3 reps) | C1 |
| rtr_rs | rtr + rs | yes | +0.59% | +0.83 / +0.61 / +0.28 / +0.65 | 21171429 | RS hurts on top of RTR |
| rtr_spf_rs | rtr + spf + rs | yes | +1.69% | +2.02 / +2.37 / +0.41 / +1.96 | 21171429 | |
| st4_rtr | rtr + ring depth 4 | yes | -4.17% | -4.12 / -4.10 / -4.20 / -4.27 | 21171429 | depth 4 adds ~0.5% |
| mc2 | K/V multicast, cluster 2 heads | yes | -1.60% | -4.96 / +0.27 / -1.94 / +0.35 | 21171430 | 64k runs per head (split): no cluster |
| mc4 | multicast, cluster 4 | yes | +2.80% | +4.10 / +0.38 / +6.47 / +0.39 | 21171430 | slower |
| mc4_all | mc4 + all-heads pass 0 in the split route | yes | +5.41% | +4.18 / +5.59 / +7.13 / +4.74 | 21171430 | |
| p0all | all-heads pass 0 in the split route (one launch) | yes (default route) | -0.32% | +0.06 / -1.05 / +0.62 / -0.90 | 21171430 | |
| rtr_spf | rtr + spf | yes | -5.46% | -5.79 / -4.99 / -5.79 / -5.27 | 21171431 | C2 |
| u3 / rtr_spf_u3 | stage loop unrolled by 3 (compile-time slots) | yes | +3008% / +2948% | | 21171431 | REJECTED: the 3 step-lambda copies are outlined (CALL, ~1 KB stack, 2900 STL); needs a non-lambda design |
| swp | rtr + spf + n-block software pipeline (exp(k) with rowsum(k-1), gemm k+1 in flight across iterations) | yes | +16.13% | +15.9 / +15.9 / +17.0 / +15.7 | 21171522 | REJECTED: ptxas injects warpgroup.wait 0 at the loop top (C7517: GMMA registers live across the back-edge), serializing every iteration |
| c1w | rtr + spf + one consumer warpgroup per CTA (64 rows), 2 CTAs / SM | yes | +2.41% | -0.09 / +4.43 / +0.97 / +4.39 | 21171654 | REJECTED: decoupling the two warpgroups of an SM does not help; the warp count is what matters |
| c1w_swp | c1w + swp | yes | +18.94% | | 21171654 | REJECTED (swp) |
| lpt | rtr + spf + grid (bh, p): global longest-first dispatch | yes | -6.06% | -8.50 / -5.15 / -5.32 / -5.21 | 21171736 | LPT adds ~-3% on B2 H16 8k |
| lpt_all | lpt + all-heads split pass 0 | yes | -6.96% | -8.56 / -7.45 / -5.80 / -6.01 | 21171736 | |
| c1w_lpt | c1w + lpt | yes | +1.13% | | 21171736 | |
| w4b | rtr + spf + 4 consumer warpgroups (256-row CTA, 112 regs, one S/dP pair, skip past-diagonal n-blocks) | yes | -8.21% | -5.53 / -9.19 / -8.81 / -9.27 | 21171573 | (first build w4 with setmaxnreg 120 hung: 4x128x(120-96) > 128x(96-24); cancelled at 4 min) |
| w4b_all | w4b + all-heads split pass 0 | yes | -10.46% | -5.36 / -13.51 / -9.85 / -12.88 | 21171573 | |
| mc2_all | mc2 + all-heads split pass 0 | yes | -4.00% | -5.03 / -4.24 / -2.97 / -3.76 | 21171530 | |
| rtr_mc2_all | rtr + mc2_all | yes | -6.69% | -7.74 / -6.79 / -5.74 / -6.47 | 21171530 | |
| rtr_st4_mc2_all | rtr + depth 4 + mc2_all | yes | -6.98% | -7.66 / -7.14 / -5.66 / -7.46 | 21171530 | |
| c3mc2 / c3mc4 | C4 stack + multicast along grid x (C = 2 / 4) | yes | -14.62% / -6.12% (in-job ref -14.42%) | | 21172253 | no gain on top of W4 (C=2 within noise), C=4 slower: REJECTED |
| w3 | C4 stack with 3 consumer warpgroups (192-row CTA, partial last CTA, 160 regs, v6 four-accumulator pipeline) | yes | -10.48% (ref -14.42%) | -11.44 / -11.47 / -8.60 / -10.41 | 21172253 | REJECTED: warp count beats intra-warpgroup pipelining |
| w5 | C4 stack with 5 consumer warpgroups (320-row CTA, 88 regs) | yes | -1.73% (ref -14.30%) | | 21172523 | REJECTED |
| stag_* | C4 stack + warpgroup start stagger (odd wg 400 / 1000 clk, wg x 300 / 700 clk), timing only | n/a | -14.31 / -13.81 / -14.13 / -14.43% (ref -15.33%, 2 shapes) | | 21172523 | REJECTED: phase alignment of the warpgroups is not the loss |

## W4 what-ifs (C4 stack, B2 H16/4 N8192, job 21172377; timing only)

| what-if | pass 0 ms | reading |
|---|---:|---|
| C4 stack | 0.555 | |
| no vector work | 0.386 | tensor + loads = the calibrated tensor floor |
| no wgmma | 0.399 | vector work alone ~ the same as tensor alone |
| no dP gemm | 0.470 | tensor work now matters (-15%) |
| no MUFU | 0.532 | |

Each half alone sits at the floor; together 1.44x. A warpgroup has one S / dP pair (112 regs), so its own gemms and
vector work only overlap during exp (dP in flight); cross-iteration pipelining makes ptxas inject a full wait (C7517,
see swp). The remaining gap is tensor / vector overlap across warpgroups.
| w4s3 | C4 stack, W4 with S0, dP0, S1 up front (3 accumulators) and dP1 after rowsum0 | yes | -2.88% (ref -13.94%) | -4.54 / -2.33 / -3.01 / -1.61 | sK1 (21172843) | REJECTED: 112 regs -> 80 B spills, ptxas C7512 wgmma serialization |
| w4p1 | C4 stack, producer = one warp after the consumer warpgroups (544 threads, no setmaxnreg) | yes | -14.40% (ref -14.13%) | | sL1 | no gain: ptxas still caps 544-thread CTAs at 96 regs (with or without -maxrregcount=120), consumers use 90 |
| w4p1s3 / w4p1s3r | w4p1 + 3 accumulators (r: -maxrregcount=120) | yes | +11.28% / +11.88% | | sL1 / sL2 | REJECTED: capped at 96 regs, 308 B spills, C7512 |

## Output-changing screens (Hao's per-row residual rule, round 17 decision)

| variant | change | geomean pass 0 (ref C4 stack) | numerics screen (DBG builds, default route, D128 matched cases + B1 H4 N8192; `p0/tools/rowres.py`) | status |
|---|---|---:|---|---|
| lazy | dP descale once per 8-key half partial (fmaf(P, dP, a), partial x dp) instead of per element: -1 FMUL / element | -17.80% (ref -13.96%), -18.30% (ref -14.40%) | gaussian cases: per-row residual changes within about +-2 x 2^-24, no exemption growth; stress cases (scale3 / scale10 / outliers): near-one-hot rows with residual ~1 in both builds move by up to 0.18 absolute; fp32 error larger on 8 (case, output) pairs (dQ / dK of gqa4, gqa3_promote, packed, bhnd; +7e-9 .. +1.3e-7 relative) | fails the strict rule as screened (fp32 error not larger on any case) |
| defer | per-lane fp64 row partials across n-blocks, one fp64 quad reduce at the end (no per-tile shuffles / fp64 folds) | -16.61% (ref -14.40%) | pending (job n2) | screening |
| lazy_defer | lazy + defer | -21.25% (ref -14.40%) | pending (job n2) | screening |
