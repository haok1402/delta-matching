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
| defer | per-lane fp64 row partials across n-blocks, one fp64 quad reduce at the end (no per-tile shuffles / fp64 folds) | -16.61% (ref -14.40%) | gaussian: within about +-1.6 x 2^-24; stress near-one-hot rows move up to 0.026 absolute; fp32 error larger on 5 pairs (<= +1.5e-6 relative on dQ of bhnd, others ~1e-8 .. 1e-7) | fails the strict rule as screened |
| lazy_defer | lazy + defer | -21.25% (ref -14.40%) | as lazy; fp32 error larger on 7 pairs (<= +1.6e-6 relative) | fails the strict rule as screened |

Reading: even DEFER, which only makes the fold more exact and keeps pass 1's per-element rounding, raises the fp32 error on
some cases by noise-level amounts and moves the ill-conditioned (near-one-hot, residual ~1 in both builds) stress rows. As
screened, the strict rule (no larger fp32 error on any case; per-row non-regression on those rows) rejects any change to
delta's rounding. These candidates wait for the rule's tolerance / degenerate-row definition (data sent to kda-sm90).


### residual-v3 check (benchmarks/numerics.py at 112de4c, default route, vs cert_v3 r17_c3 captures; job kda-p0-n3)

Per-row regressions / exempt (accepted exempt) per D128 matched case, worst fp32-error ratio vs prev:

| candidate | gaussian cases (min, gqa4, gqa3_promote, mqa, packed, bhnd, heldout, constk) | stress (scale3, scale10, outliers) | exempt growth | worst fp32 error vs prev | determinism |
|---|---|---|---|---|---|
| DEFER | 0 regressions | 37 / 9 / 31 | none | +3 ppm (outliers dK), +1.5 ppm (bhnd dQ) | pass |
| LAZY | 3 (gqa4) + 2 (gqa3) + 2 (mqa) + 1 (heldout) | 43 / 136 / 63 | none | +2.0 ppm | pass |
| LAZY+DEFER | as LAZY | 62 / 141 / 72 | none | +1.7 ppm | pass |

(The identity FAILs in the report are my unstamped what-if builds.) Status: all three wait for Hao's tolerance /
degenerate-row decision; DEFER is the cleanest (no regression outside the ill-conditioned stress rows).

| w4u3 | W4 stage loop unrolled by the ring depth via a macro body (constant slots) | yes | -14.18% (ref -14.05%) | | sN1 | REJECTED: 540 vs 557 instr / stage, no time gain |

## Final summary (helper kda-pass0)

Accepted into kda-sm90 from this branch: C1 RTR, C2 SPF (with C1), C3 LPT + W4 (+ fallback). The integrator's gate
measured pass 0 -5.21% for C2 and -7.67% for C3 on top of C2. Their sol4-v2 table puts the C2 build at 62.0% of the
37.1 ms tensor floor; C3 brings that to ~67%.
READY, waiting on tooling: C4 P0ALL (b864956, split.py; -14.35% / -14.47% for the C1..C4 stack vs r14_h3). It is
byte-equal on all four routes; equivalence.py needs the all-heads delta schema. My estimate: ~69% of the floor.
Waiting on Hao's numerics tolerance: DEFER (-2.2% on top of the stack), LAZY (-3.9%), LAZY+DEFER (-6.9%). Above are the
residual-v3 numbers.

What bounds pass 0 now (W4 what-ifs): tensor-only and vector-only each run at the calibrated floor; the full kernel is
1.44x. The loss is overlap between the 4 warpgroups. Each warpgroup holds one S / dP pair (112 regs), so its own gemms
overlap only with exp. Cross-iteration pipelining is serialized by ptxas (C7517), and a third accumulator spills (C7512).
The 544-thread producer-warp layout is still capped at 96 regs.

Levers, each tried in at least two designed variants:
- pipeline depth / prefetch: ring depth 2 (+3..8%), depth 4 (-0.5%, and -0.3% on W4); scale prefetch SPF (accepted);
  global LPT dispatch (accepted).
- instruction cuts (bitwise): RTR (accepted), SPF (accepted), U3 lambda (outlined, 30x slower), W4U3 macro (no gain).
  Output-changing: LAZY, DEFER (pending the rule).
- tile shape / occupancy: W4 (accepted), W3 (-10.5%), W5 (-1.7%), C1W 2 CTAs/SM (+2.4%), RS Q/dO in registers
  (-1.65% alone, slower with RTR), W4S3 (spills), W4 producer warp (no gain), multicast C=2 / 4 (-1.6% / +2.8% alone,
  nothing on top of W4).
- overlap / scheduling: SWP (+16%, C7517), C1W+SWP (+19%), warpgroup stagger (no gain).
- fusion / sharing with forward / pass 1: not buildable without breaking the identity.
  - Delta must be complete per row before pass 1 forms dS and psi, and pass 1 is column-CTA.
  - Reusing pass 0's P / dP needs 2 x B H N^2 / 2 fp8 bytes and fp8-rounded dS inputs.
  - The forward has no dO.
  - Sharing done instead: K / V across GQA heads (multicast, no gain on W4), and one all-heads launch (C4).
- Not tried: a joint pass-0 + pass-1 exp-line change (fma(s, dsc, 8 - lse), -1 FADD per element in both passes). It is
  only legal if pass 1 changes identically.

To reach 90% of the floor, the next steps lie outside bitwise pass-0 scheduling: the numerics tolerance for DEFER / LAZY,
or the joint exp-line change with pass 1.

## Round after residual-v4 (branch kda-sm90-pass0-num, on the accepted head 3106faf, no C4)

Commits (ungated, named numerics changes): e3275a3 P0DEFER, c49c0d6 P0LAZY (on DEFER), 4dc4268 P0FOLD (exp argument
as one fmaf(s, dsc, 8 - lse), on LAZY). Screen variants: e881f20 P0NORM on FOLD, a23f011 P0NORM on DEFER
(delta = sum P dP / sum P). Note: `p0/tools/{defer,lazy}.patch` are macro-gated what-if patches; applied without
-DKDA_P0DEFER / -DKDA_P0LAZY they compile the unchanged kernel. kda-sm90's jobs 21176380 / 21176381 measured exactly
that (~0%, trivial PASS).

Stamped builds of the commits vs frozen r17_p1 (ab.py paper / pass0, one job each):

| commit | pass 0 geomean | decision (this job) |
|---|---:|---|
| e3275a3 DEFER | -3.04% | REJECT (one shape slower beyond spread) |
| c49c0d6 LAZY + DEFER | -7.50% | ACCEPT |

Screen (job sO1, 2 reps, vs r14_h3): c49c0d6 -19.27%, + FOLD -23.73%.

residual-v4 evaluation (`p0/tools/v4.py` on numerics.py captures of the stamped builds, all four routes, vs cert_v3 r17_c3):

| commit | well-conditioned row regressions | ill-conditioned rows (accepted > 1e-3) | mean dQ error | strict v4 |
|---|---|---|---|---|
| DEFER | 28 / route (all d128_scale3, worst 0.026 absolute) | 8124, worst move 0.11 | +1.35e-9 | FAIL |
| LAZY + DEFER | 57 / route (gqa4 3, gqa3 2, mqa 2, heldout 1, the rest scale3) | 8124, worst move 0.18 | +3.4e-9 | FAIL |

Reading: the regressing rows have a small accepted residual by chance. A row's residual is dominated by
sum_j P_j != 1 (delta = sum P dP satisfies sum_j P_j (dP_j - delta) = 0 only when sum_j P_j = 1), so any rounding change
of delta moves them. Most defensible next step, protecting the zero-sum identity: P0NORM, delta = sum P dP / sum P,
which makes the identity exact up to the rounding of the sums. Its numerics and timing are in flight.

### FOLD and NORM (residual-v4, default route; stamped builds vs frozen r17_p1)

| commit | pass 0 geomean vs r17_p1 (ab.py, 1 job) | v4 well-cond. regressions | degenerate growth | mean dQ / dK error | rows better / worse vs accepted | v4 |
|---|---:|---|---|---|---|---|
| 4dc4268 FOLD (on LAZY + DEFER) | **-13.12%** ACCEPT | 78 (gqa4 5, gqa3 6, mqa 2, heldout 11, bhnd 3, min 1, packed 1, stress 49) | 0 | +7.5e-9 / +7.6e-9 | 62707 / 93453 | FAIL |
| e881f20 NORM on FOLD | -4.12% | 9 | 2 (scale10) | +3.0e-8 / -9.2e-8 | | FAIL |
| a23f011 NORM on DEFER | +5.74% | 6 | 2 (scale10) | +2.3e-8 / -9.6e-8 | 135722 / 32937 (mean residual 1.56e-2 -> 1.29e-2; ill rows 2444 better / 321 worse) | FAIL |
| e3275a3 DEFER | -3.04% | 28 | 0 | +1.4e-9 / | 49284 / 46267 | FAIL |
| c49c0d6 LAZY + DEFER | -7.50% | 57 | 0 | +3.4e-9 / | 67368 / 64696 | FAIL (also kda-sm90 job 21177096) |

Decision (most defensible under Hao's residual-v4 as written; recorded here, not a rule change): every change to the
rounding of delta moves roughly half the rows either way. Even delta = sum P dP / sum P, which makes the zero-sum
identity exact and improves 4x more rows than it worsens, regresses a handful of well-conditioned rows beyond
k 2^-23 and raises the mean dQ error by ~2e-8. Strict per-row non-regression therefore rejects all pass-0 numerics
changes. The numerics path for pass 0 is closed under v4; FOLD's -13% (the largest single pass-0 gain found) and NORM
(the identity-exact delta) stay documented for a future rule. Pass-0 work continues on bitwise levers only.

## Status (end of the residual-v4 round)

- Accepted into kda-sm90 from this helper: C1 RTR, C2 SPF, C3 LPT + W4 (+ fallback). Pass 0 ~66% of its sol4-v2 floor
  (kda-sm90 round-17 report: 65.9%).
- READY, waiting on kda-sm90 tooling: C4 P0ALL (kda-sm90-pass0, b864956; equivalence.py all-heads delta schema).
- Closed under strict residual-v4: DEFER, LAZY, FOLD, NORM. Measured: FOLD stack -13.1%, the largest pass-0 gain found.
- Bitwise levers exhausted with measurements:
  - Structure: W4's tensor-only and vector-only times each sit at the floor; the remaining 1.44x is overlap.
  - Registers: one S / dP pair per warpgroup at 112 regs, 3-accumulator variants spill.
  - ptxas: cross-iteration pipelining gets C7517 full waits.
  - Instruction trims: unrolling (W4U3, -17 instr / stage) gains nothing.

## C5: NML + register split 104 / 24 (bitwise, stacking candidate) — branch kda-sm90-pass0-b2, e61c60d on kda-sm90 9be3622

Port of kda-sm90's S17-p0 'nomaskloop' onto W4: leading stages whose n-blocks need no mask run a loop without the mask
test / skip predicates. Its other pieces are already in W4: 'skipdead' (W4's nb_last skip) and the RTR scatter (C1).
The register split was re-swept on W4: 112/24, 112/32, 104/24, 104/40, 96/24, 96/40, with 104/24 best.

| evidence | result |
|---|---|
| screen sQ1 (vs r14_h3, 2 reps) | ref -12.57%, NML -13.54%, NML + 104/24 -13.82%, 112/32 -13.58%, 104/40 -13.61% |
| screen sR1 (on e61c60d) | 104/24 -13.53%, 96/24 -13.01%, 96/40 -13.05%, 112/24 -12.98% |
| equivalence (stamped build) | BITWISE EQUAL vs frozen r17_p1 on default / kv / head / streams2, 72 cases each (jobs 21178004, 21178009) |
| ab.py paper / pass0 vs r17_p1 | **-1.27%** (job 21178004; -1.07 / -2.09 / +0.17 / -2.08) and **-1.42%** (job 21178009; -0.82 / -2.02 / -0.89 / -1.95); below 3%, no shape slower beyond spread |
| resources | launch 96, consumer 104, producer 24; STACK 0, no spills |

Status: **READY for stacking** (bitwise, >= 1% in two jobs).

## Stack C5 + C4 (+ tooling) for kda-sm90 c1ceece — branch kda-sm90-pass0-stack (3766b4f)

Commits on c1ceece: f254c01 (C5: NML + 104/24), 6061e7b (C4: P0ALL, split.py), 3766b4f (tooling:
benchmarks/equivalence.py v4).
- The tooling commit canonicalizes the matched delta captures to all.delta_fp8.{lse,out}: the delta_fp8 calls are
  concatenated in head order, checked to tile [0, H) exactly. Manifest version 3 -> 4.
- With it, a per-group build (r17_fwd) and the all-heads build (C4) compare byte for byte.

| evidence | result |
|---|---|
| equivalence (v4 tooling, stamped build) | BITWISE EQUAL vs frozen r17_fwd on default / kv / head / streams2, 72 cases each, in two jobs (21178884, 21178885) |
| ab.py paper / pass0 vs r17_fwd | -3.30% (21178884, ACCEPT), -2.86% (21178885, REJECT: B2 H32 8k at -0.06%), -3.29% (21179368, ACCEPT), -3.26% (21179369, ACCEPT); per shape about -1 / -5.1 / -1.2 / -5.4% |
| resources | consumer 104 / producer 24 at launch 96, STACK 0, no spills |

Status: **READY** (>= 3% in the two most recent fresh jobs; all four jobs disclosed).

Also screened on this stack (sT2, 3 reps, vs r17_fwd; stack ref -3.05%): Q/dO EVICT_FIRST -3.42%, K/V EVICT_NORMAL
-3.69%, both -3.63%, ring depth 4 -3.90%. All bitwise; all within the 8k H32 shape's repetition range (up to 0.044 ms),
so not taken. PERS (persistent W4, snake LPT tiles, next-tile K/V prefetch): -9.46% / -9.57% vs C5's -13.64% / -13.89%
(spills in the tile-loop state; per-tile Q/dO hand-off): REJECTED after two variants (104 and 112 consumer regs).

## Accepted: pass-0 stack (kda-sm90 2fe6079)

f254c01 (C5 NML + 104/24) and 6061e7b (C4 P0ALL) were cherry-picked; equivalence v4 tooling 3766b4f was adopted as
853f7e6. Gate 21179675: bitwise on all four routes, pass 0 -3.19%. Confirmation 21179800: -3.43%. New accepted root:
.local/frozen/r17_p0s. Byte references for bitwise gates: the r17_fwd v4 dumps.

Accepted from this helper so far: C1 RTR, C2 SPF, C3 LPT + W4 (+ fallback), C4 P0ALL, C5 NML + 104/24.
