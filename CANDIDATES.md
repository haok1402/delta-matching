# kda-fwd: forward (D128) candidates

Helper kda-fwd, branch `kda-sm90-fwd`. Target: the D128 forward (`fwd_fp8`), causal matched, acceptance shapes
B2 N8192 and B1 N65536 at H/Hk 16/4 and 32/4. A = the latest accepted frozen build (`.local/frozen/r14_h3`).
KSPF (k-scale prefetch) belongs to kda-sm90 and is not tried here.

## Design diff against FA3's fp8 forward (D128)

| | FA3 fp8 | ours (r14_h3) |
|---|---|---|
| tile | 128 x 224, 2 consumer WGs + producer, 2 stages | 128 x 256 (BN256), same WGs / stages |
| intra-WG overlap | QK(n) and PV(n-1) issued back to back; softmax(n) under PV(n-1) | none (PV-first: tSrS 128 + tOrD 64 + tOrO 64 + P 32 = 288 regs would not fit 240) |
| PV | one gemm into tOrO | zero-init temp, 4 x (2-atom sub-gemm, wait, 64-FMUL rho rescale), promote 64 FFMA |
| per-step vector work | ~1 FFMA + 1 MUFU + 1 FMNMX + 1 FADD per score | + 1 FMUL S descale per score, 3 x 64 rho FMUL, 64 promote FFMA, 4-group max / fold / rho bases |
| ping-pong | scheduler barrier around QK + PV issue | scheduler barrier around the PV chain + QK issue; the WG's own rescale bubbles leave the tensor pipe idle |

Earlier rejections that bound this search: PVIL (round 7, interleaved half-width temps: -0.08% D128, chain length
unchanged), GRPFOLD (round 8, four independent half-width group folds: D128 slower), both because they keep the same
per-element op count and the same chain length per temp.

## Screening ledger (every variant; forward kernel ms summed over the four shapes; quick screens)

Variant switches live in the what-if commits (`KDA_FWI_*`, `KDA_FLAGS_OVERRIDE`); they are not candidates.

| # | tag | lever | what it changes | outputs | screen result |
|---|---|---|---|---|---|
| S1 | v0 | control | what-if source, no switch | bitwise A | A-equal (O sha same), -0.05% (job 21171358) |
| S2 | bn128pf | tile | BN 256 -> 128, PV-first | order class | +9.97% (job 21171433, 4 shapes, 2 reps); REG 168, STACK unchanged |
| S3 | bn128fa | overlap | BN 128 + the existing FA3-order step (QK then PV chain, softmax partly under g1) | order class | +8.65% (21171433); same O as S2 (as expected) |
| S4 | ovl3 | overlap | BN 128 + new FA3-order step: g0, QK(n), rho under QK, g1, whole softmax under g1 | order class (= S2 / S3) | +16.3% sum (21172009); O equals S2 (correct, slow: g1 queues behind the other WG's gemms under the ping-pong) |
| S5 | ovl3s3 | overlap+stages | S4 with 3 stages | = S4 | +15.9% (21172103) |
| S6 | bn128pfs3 | stages | S2 with 3 stages | = S2 | +10.54% (21171744, shapes 16/4 8k+64k) |
| S7 | ovl3ns | overlap | S4 without the inter-WG ping-pong barrier | = S4 | not screened (S10 shows the ping-pong is worth ~8%) |
| S8 | dfold | S descale | max on raw codes (bitwise ms), code argument fma(x, g*l2, -ms) (no FMUL per score) | round-off | **-3.92%** (21171743, 16/4 8k -5.4%, 64k -3.9%); O differs (round-off); resources unchanged |
| S9 | askip | alpha | skip O *= alpha when alpha == 1 for the whole warp | bitwise | +33.8%: the warp-vote branch spills (D128 STACK 24 -> 168 / 176 B) and changes FMA contraction (O differs): dead |
| S10 | nosched | overlap | BN256 without the ping-pong barrier | bitwise | +8.65% (21171743): the ping-pong barrier is worth ~8%; bitwise |
| S11 | abnorho | ablation | no rho rescale (wrong numerics; sizing only) | wrong | -8.63% (21171744): the rho rescale costs ~8.6% of the D128 forward on these shapes |
| S12 | abnodesc | ablation | no S-descale multiply (wrong numerics; sizing only) | wrong | -4.83% (21171744): the S-descale multiply costs ~4.8%; DFOLD recovers 3.9 of it |
| S13 | split2 | rho | BN256: two full-width temps (groups 0-1, 2-3) interleaved, rescales under the other temp's gemm, 2 promotes | order class | +8.4% (21172009); D128 STACK 16/24 -> 56/40 B; the interleaved temps do not pay |
| S14 | rhopre | rho | the three rho computed before the PV chain | bitwise | -0.35% (21171744), bitwise (same O) |
| S15 | ovl3df | combo | S4 + S8 | round-off | +5.7% (21172009) |
| S16 | ovl3as | combo | S4 + S9 | = S4 | not screened |
| S17 | dfas | combo | S8 + S9 | round-off | not screened: built with the S9 spill (STACK 176 / 192 B) |

| S18 | qks2 | overlap | QK(n) in two 128-key halves, half 0 issued inside the PV chain | bitwise | +33.4% (21172103): D128 STACK 24 -> 216 / 240 B (P codes still live) |
| S19 | qks4 | overlap | QK(n) in four 64-key quarters, two inside the PV chain gaps | bitwise | +28.1% (21172103): STACK 176 / 184 B |
| S20 | qks4df | combo | S19 + S8 | round-off | +29.0% (21172103) |
| S21 | lazya | alpha | tOrO in a lagging reference basis, per-tile O *= alpha dropped (rare warp-uniform rebase) | round-off (O only; LSE / dQ / dK / dV bitwise) | -1.37% sum / -1.09% geo (21172182); first version NaN at x10 / x30 logit scales (rebase after the promote): fixed by rebasing before it |
| S22 | lazydf | combo | S21 + S8 | round-off | **-5.95% / -6.15% geo** (21172182) |
| S23 | ovl3b | overlap | S4 with g1 issued inside the WG's ping-pong turn | = S4 | +10.9% (21172182) |
| S24 | bn128ld | tile | S2 + S21 + S8 | round-off | +8.2% / +6.3% geo (21172383): 128-key tiles lose even without alpha |
| S25 | ovl3ld | combo | S4 + S21 + S8 | round-off | +1.4% / +0.9% geo (21172383) |
| S26 | ldrp | combo | S21 + S8 + S14 | round-off | **-6.74% / -7.31% geo** (21172383); -6.93% geo (21172654) |
| S27 | abnoexp | ablation | S26 without exp2 (wrong) | wrong | -16.6% geo vs A (21172654): exp2 / MUFU costs ~10% beyond S26 |
| S28 | abnomax | ablation | S26 with one element instead of the group max (wrong) | wrong | -7.9% geo: the max costs ~1% |
| S29 | abnorho2 | ablation | S26 without rho (wrong) | wrong | -10.9% geo: rho still costs ~4% after RHOPRE |
| S30 | vcol / vcol3 | V layout | read V column-major (no producer transpose), 2 / 3 stages (timing only) | wrong | +3.6% / +3.8% geo (21173012): the consumer P permute (permute_Aregs_fp8) costs more than the producer transpose saves; fork's V_colmajor path needed 3 compile repairs |
| S31 | qkpre0 / qkpreld | overlap | after the last PV retire (P dead) issue QK half 0 before the promote (O 64 + D 64 + S0 64 live), half 1 into D's registers | bitwise / round-off | +33.7% / +18.0% geo (21173012): STACK 176-224 B (spills despite the 192-register ledger) |
| S32 | poly8 / 4 / 2 | MUFU | S26 + every 8th / 4th / 2nd code exp2 by a degree-5 FMA-pipe polynomial (1.5 ulp) | round-off | +2.6% / +3.3% / +22.8% geo vs A (S26 -6.8%) (21173009): MUFU throughput is not the limiter; extra issue slots cost more |

Reference (job 21171433): FA3 fp8 forward on the four shapes 57.617 ms vs A 74.572 ms (A = 1.294 x FA3); per shape
0.572 / 1.131 / 18.374 / 37.539 ms vs 0.783 / 1.542 / 23.783 / 48.464 ms.

SASS of the steady-state (unmasked) D128 causal loop of A: 1288 instructions per 64 x 256 step, FMUL 411 (S descale
128, rho 192, alpha 64), FFMA 212 (exp 128, promote 64), FMNMX 164, MUFU 149, FADD 138, F2FP 64, QGMMA 12.

| S34 | probe0 / probeld | diagnostic | clock64 phase sums per WG step (printf from CTA 0/1; spills, timing inflated) | n/a | per WG step ~5400 cy: ping-pong barrier wait ~2%, gemm turn (PV chain + promote + QK issue) ~2450, softmax ~2700 (job 21174714): each WG is a latency-bound serial chain; the WGs rarely wait for each other |
| S35 | bm192 | structure | 192 x 128 tiles, three consumer WGs (160 regs), PV-first BN128 | order class | +1.6% geo (21175086) |
| S36 | bm192ld | structure | S35 + LAZY + DFOLD | round-off | **-12.2% geo** (21175086); NaN at x10 / x30 (fully masked rows in the first tile reset the lazy reference to 0; rebase factor 2^(0 - m) -> inf * 0): fixed by clamping the factor to 1 |
| S37 | bm192s3 / b3lds3 | structure | S36 with 3 stages | round-off | -12.5% (21175086), -13.1% geo (21175696) |
| S38 | b3ldns | structure | S36 without the ping-pong barrier | round-off | -1.1% geo (21175696): the barrier still matters with three WGs |
| S39 | bn128l / bm192l / bm192d | bisect | lazy / dfold alone at BN128 | | used to locate the S36 NaN |

| S40 | probeb3 | diagnostic | phase probe on the C3 configuration | n/a | per WG step ~4600-4800 cy; gemm turn ~2750 (60%), softmax ~1650 (job 21176432) |
| S41 | b3il | rho | C3b + the two 32-register head-dim slabs interleaved (IL2: each slab's rescale under the other's sub-gemm) | = C3 | -12.60% geo vs C3b -12.88% (job 21176682): no gain |
| S42 | b3arv | ping-pong | C3b with the turn ending after the PV chain issue (QK outside the turn) | = C3 | -3.00% geo vs C3b -12.47% (21176953): QK must stay in the turn |
| S43 | b3tt2 / b3tt3 | ping-pong | C3b with the step split into 2 / 3 ring turns (all waits outside turns) | = C3 | -0.31% / -0.60% geo vs C3b -12.89% (21177145): extra ring turns cost far more than the bubbles |

| S44 | b3qkr / qkr256 | ping-pong | only the QK issue inside the ring turn, PV chain free-running (3 WGs / 2 WGs) | = C3 / = S26 | -3.20% / +3.10% geo (job 21177775; C3b -12.79%, S26 -6.89%): the turn must cover PV chain + QK |

| S45 | b3os / b2os | registers | O accumulator in shared memory (thread-private float4 chunks), FA3-order step (turn = g0 + QK, rescale / g1 / softmax / promote outside) | = b3ld / bn128ld class | b2os (2 WGs, BN128) +21.7% geo (job 21178384); b3os (3 WGs) does not launch: O 96 KB + 2 K/V/Vt stages + Q + epilogue exceed 227 KB. Rejected |

| S46 | b3kw | ping-pong | C3b with the K pipeline wait moved ahead of the ring turn | = C3 | +30.1% geo (job 21178438 build: STACK up to 384 B, also on d256): rejected |
| S47 | b3n64 / b3n64o / b2n64o | tile | 64-key tiles (one P-hat group: no rho), 3 WGs PV-first / 3 WGs FA3 order / 2 WGs FA3 order, 4 stages | order class | -3.44% / +62.2% (spills) / +16.8% geo vs C3b -12.64% (job 21178606): per-tile overhead dominates; 128-key tiles with 3 WGs are the sweet spot |

| S48 | b3pre | turn | lazy rebase, rho and promote weight computed before the ring turn | bitwise (O = b3ld) | -1.71% geo vs r17_fwd (job 21178684 / s22) -> C4 |
| S49 | b3ktok | turn | + K try-wait token before the turn | bitwise | -1.85% screen; as C4b in A/B -0.58% / -0.75% (below C4): rejected |

| S50 | c4v | scale loads | C4 + this tile's v-scales loaded into temporaries at the step start (VPF) | bitwise | +0.08% geo vs r17_fwd (C4 -1.04%, job s25): cancels C4's gain; rejected |
| S51 | c4s4 | stages | C4 with 4 K/V stages | bitwise | -0.98% geo (C4 -1.25%, job s26): no gain |

| S52 | c4m | masking | C4+SMSC: masked-range tiles fully visible for a WG run the unmasked step body (second inlined step instance) | intended bitwise | +18.1% geo (job s28): causal D128 STACK 16 -> 136 B, d256 1408 B; O differs (prologue-masked rows): rejected |
| S53 | c4m2 | masking | C4+SMSC: one step body, the causal mask skipped by a warp-uniform branch on fully visible tiles | bitwise (O = r17) | +20.0% geo (job s30): STACK 144 B (d256 312 B); a branch in the step breaks the allocation: rejected |

| S54 | host | host path | fwd_fp8 host cost (integrator lever: ~0.3 ms per call in paper C.2) | n/a | not a lever: host 0.015 ms per call, back-to-back wall 0.684 ms = kernel (r17_p0s, B2 N8192 16/4, job 21180725, nsys cuda_api_sum in runs/host); the C.2 per-call gap is idle-GPU launch latency in the event window |

## Candidates

### C1 FWDLD + RHOPRE (current commit c3df077 on kda-sm90-fwd, on a371616; flags FA3FP11_HD128_FWDLAZY=1, FA3FP11_HD128_FWDDFOLD=1; builds `kda-work/frozen/c1e`, `c1e_dbg`) : READY (timing); numerics under residual-v3: see below

- What it changes (D128 PV-first forward only; d256 untouched; new certified flag `FA3FP11_HD128_FWDLD=1`):
  1. S descale folded into the code argument (group maxima on raw codes and descaled once per row / group: gm, ms,
     row_max, alpha bitwise; argument fma(x, RN(q8sc k8sc l2), -ms) instead of fma(RN(x q8sc k8sc), l2, -ms));
  2. lazy O rescale: tOrO kept in a lagging per-row reference (rebased, warp-uniform, before the promote, only when the
     running max drifts > 2^8 from it); the per-tile tOrO *= alpha is gone; finalize folds the reference into 1/l;
  3. RHOPRE: the three rho of the PV chain computed before it (bitwise on its own).
- Outputs: NOT bitwise (named numerics change, round-off class). O changes (lazy + fold); LSE changes at round-off (the
  row sum takes the re-rounded p of the fold), hence dQ / dK / dV change at round-off too. Per-row evidence
  (`kda-results/kda-sm90-fwd/n2`, fp32 reference, TF32 off, job 21172655): O relative L2 vs fp32 equal to A within 6 digits on
  all six cases (randn x1 / x3 / x10 / x30 logit scale, B1 H16/4 N8192, B2 H4 N2048, B1 H4/1 N16384); per-row error
  differences symmetric (e.g. B1 H16/4 N8192: 39 of 131072 rows worse by > 1e-4 abs, max +1.07e-3, min -1.95e-3 =
  e4m3 code flips at rounding boundaries); no NaN / Inf (the first lazy version gave NaN at x10 / x30; fixed).
- Resources: REG 168, STACK / LOCAL unchanged on every forward symbol (D128 24 / 24, LOCAL 0); no spill / C7510 /
  serialization diagnostics. Steady-state loop 1288 -> 1191 static instructions (FMUL 411 -> 295 static incl. the rare
  rebase branch).
- A/B (benchmarks/ab.py run --grid paper --target fwd, A = r14_h3, 3 reps, two fresh jobs):

| job | fwd geo-mean | fwd sum ms | e2e fwd+bwd | shapes slower beyond spread | decision |
|---|---:|---:|---:|---|---|
| 21172873 (`kda-results/kda-sm90-fwd/ab_c1c_1`) | **-7.05%** | 75.682 -> 70.270 (-7.15%) | -0.92% | none | ACCEPT (this job) |
| 21172874 (`kda-results/kda-sm90-fwd/ab_c1c_2`) | **-6.79%** | 75.229 -> 70.060 (-6.87%) | -1.65% | none | ACCEPT (this job) |

- Still needed for acceptance (kda-sm90): `gate.py --mode numerics` (four routes, DBG pairs) under Hao's residual
  contract; pytest on the build.

- Re-measured on the current accepted build r17_c3 (rebased commit c3df077, build c1e): fwd geo-mean **-6.86%**
  (job 21174360) and **-6.57%** (21174361); no shape slower beyond spread; ACCEPT (this job) in both.
- residual-v3 screen, default route (kda-sm90's r17_c3 base / acc / dbg captures and defects manifest; job 21174390,
  `kda-results/kda-sm90-fwd/nv3_default`): (1) 1 matched row above its accepted residual by more than round-off
  (d128_mqa/causal/matched, group 0) of 334592; (2) exemptions 8378 = accepted (no growth); (3) deterministic; error vs
  fp32 over 288 case / output pairs: 43 up (median 1e-6 rel, max 3.9e-5), 53 down (min -6.2e-5), 192 equal ->
  strict "not larger" FAILS on noise (90 problems). Needs Hao's ruling on fp32-noise tolerance and the one row.

### C2 LAZY + RHOPRE (commit 76513de on kda-sm90-fwd-c2: C1 with FA3FP11_HD128_FWDDFOLD=0) : REJECTED (below 3%; kept as the numerics-safe fallback)

- O-only numerics change in matched mode: LSE and the matched backward are bitwise (stale mode's delta reads O).
- A/B vs r17_c3: fwd geo-mean -2.55% (21174362), -2.35% (21174363): below the 3% target; not bitwise, so not stackable.
- residual-v3, default route (job 21174390): (1) 0 regressed rows of 334592; (2) exemptions 8378 unchanged;
  (3) deterministic; fp32 error 22 up (median 1e-6, max 5.7e-6) / 22 down / 244 equal -> fails only the strict "not
  larger" clause (44 noise problems).
- Resources (both C1 and C2): REG 168, STACK / LOCAL unchanged, no spill / C7510 / serialization diagnostics.

## Screens after the C1 hand-off

| # | tag | lever | what it changes | outputs | screen result |
|---|---|---|---|---|---|
| S33 | qkh | QK latency | S26 + QK issued as two 128-key halves at the same point; group max of half 0 under half 1 | = S26 | -3.09% geo vs A (S26 -6.65%, job 21174063): splitting the QK wgmma costs more than the overlap gains |

### C3 BM192 (branch kda-sm90-fwd-c3: bbbef1d = C1 + BM192; 47d72e3 = + 3 stages (C3b)) : READY (timing); numerics per the recorded v4 reading below

- What it changes (on C1): `FA3FP11_HD128_BM192=1` (new certified flag) and `FA3FP11_HD128_BN256` 1 -> 0: the hd128
  forward runs 192-row m-blocks over 128-key tiles with three 64-row consumer WGs (setmaxnreg 160 / 32, FA3's
  three-WG scheduler barrier); each WG reads the Q scale of its 128-row block; partial last m-blocks via FA3's row
  bounds. C3b adds `FA3FP11_HD128_FWDST=3` (K/V stages 2 -> 3; bitwise vs C3). Also: the lazy rebase factor is clamped
  to 1 (bitwise for live rows; needed when a 128-key tile leaves rows fully masked).
- Why: the phase probe (S34) shows each WG is a latency-bound serial chain; a third chain in flight raises tensor-pipe
  occupancy. Alone (S35) it is neutral; with the C1 instruction cuts the two compound.
- Screen (16/4 8k + 64k, 2 reps, job 21175696): C3 -12.61% geo, C3b-equivalent -13.13%, C1 -6.81% on the same job.
- Numerics (fp32 reference, job 21175697, `kda-results/kda-sm90-fwd/n5`): no NaN / Inf at x1..x30; O relL2 change per case -6.4e-5 ..
  +2.5e-5 relative (absolute -3e-6 .. +6e-6), e.g. B1 H16/4 N8192 4.733676e-2 -> 4.733682e-2. Per-row differences
  symmetric (order class: two 64-key groups per promote).
- Rule reading recorded (Hao: decide and record): residual-v4's "per case, the relative error may rise by at most
  1e-5" is read as an ABSOLUTE rise of the relative-L2 metric (C3's largest is +6e-6 at x30), and the mean over cases
  per output must not rise.
- Resources: D128 forward symbols REG 128 at launch (setmaxnreg 160 for consumers), STACK 16 B, LOCAL 0; no spill /
  C7510 / serialization diagnostics.
- A/B (ab.py --grid paper --target fwd vs r17_c3, two fresh jobs, build `frozen/c3` = bbbef1d): fwd geo-mean
  **-12.88%** (job 21176300) and **-13.20%** (21176339); fwd sum 75.563 -> 65.862 and 76.553 -> 66.142 ms; no shape
  slower beyond spread; ACCEPT (this job) in both. (C3b = + 3 stages screened -0.5% further; A/B pending.)
- Phase probe on the C3 configuration (S40, job 21176432): per WG step ~4600-4800 cycles, gemm turn ~2750 (60%),
  softmax ~1650: the PV chain + QK turn is now the long phase (next levers target it).
- Numerics, default route vs the r17_c3 captures (job 21176433, `kda-results/kda-sm90-fwd/nv3c3`, residual-v3 checker):
  rows: 1 matched row above its accepted residual (d128_mqa/causal/matched group 0, the same near-one-hot row as C1),
  exemptions unchanged; deterministic. Error vs fp32 (absolute change of the relative-L2 metric, 288 case / output
  pairs): matched cases max rise +8.3e-6 (all <= 1e-5), finite-case mean change matched O +0.0004%, dQ / dK +0.0001%,
  dV -0.0000%; stale / full cases: 10 dQ / dK cases rise > 1e-5 (max +2.1e-4, mqa stale dK) with up / down counts
  balanced (dK 7 up / 13 down, dQ 11 / 9) and the finite mean of stale dQ falling.
- Rule reading recorded (Hao: helpers decide, record with evidence): the per-case 1e-5 bound and the mean
  non-rise are applied to the matched-mode cases (the method), with "mean non-rise" read as "not above fp32
  round-off of the mean" (|change| <= 1e-6 absolute); stale / full-attention dQ / dK move by the conventional
  delta rowsum(dO * O) amplifying O's round-off into fp8 dS' code flips (sign-random: balanced counts, the same
  pattern at smaller amplitude in the O-only C2) and are reported, not gated. kda-sm90 owns the gate and may
  apply the stricter reading; under it C3 fails on 10 stale-mode cases and C1 on 4.

#### C3 at the residual-v4 gate (kda-sm90 job 21176828: FAIL) and the follow-up (kda-fwd)

- Gate failures on the default route: (1) zero-input NaN pattern moved (522752 vs 522240 NaN), (2) 70 well-
  conditioned rows above accepted + k 2^-23, (3) per-output mean error +~3e-7 absolute.
- Fix for (1), commit 3565487 (C3f): the folded scale q8sc * k8sc floored at FLT_MIN (bitwise for nonzero scales);
  d128_zeros O now has 0 NaN and is finite (O = 0 = the fp32 reference for V = 0). The defect is removed; no 192-row
  tiling can reproduce the frozen NaN pattern.
- Evidence on (2), `kda-results/kda-sm90-fwd/ctl`, jobs 21177451 / 21177475: the ACCEPTED r17_p1 DBG build with only its LSE moved by
  +1 ulp (no kernel change) "regresses" 1930 well-conditioned rows under the same rule (d128_scale10 up to +0.29,
  outliers +0.24, scale3 413 rows, d256 stress cases too); -1 ulp likewise. C3f: 70 up / 180 down; C3n (C3 without
  FWDDFOLD, b1613d8): 23 up / 14 down. Against a per-row envelope k 2^-23 + max over the +-1 ulp LSE controls of
  the accepted build: C3n 0 rows beyond, C3f 1 row (mqa, excess 2.5e-7).
- Rule reading proposed (recorded per Hao's autonomy rule; the gate owner decides): the fp32 round-off allowance
  of a row includes the propagated round-off of its forward input LSE, measured by the +-1 ulp LSE controls of the
  accepted build. Without it every forward change that moves LSE at round-off fails, whatever its accuracy.

### C3n (kda-sm90-fwd-c3n b1613d8: C3f without FWDDFOLD) : READY (timing); numerics: 0 rows beyond the LSE-ulp envelope

- A/B vs r17_p1: fwd geo-mean -5.82% (job 21177596), -5.88% (21177597); no shape slower beyond spread.
- C3f (3565487) vs r17_p1: -12.81% (21177598); the fold carries ~7 points on top of C3n.
- Strict v4 rows: 23 up / 14 down (C3f 70 / 180); beyond the +-1 ulp LSE envelope: 0 (C3f 1, excess 2.5e-7).

### C3g (kda-sm90-fwd-c3g b0db717 on 9be3622) : ACCEPTED by kda-sm90 as c1ceece (numerics gate 21178215 all four routes, fwd -12.89%; confirmation 21178561 -12.68%)

- C3f + C3n2's zero-scale floor, with FWDDFOLD applied on UNMASKED steps only (the first tile and masked / diagonal
  steps keep the exact descale). Reason: the one row that kept C3f out (d128_mqa group 0, row 4098) has 3 visible keys;
  its softmax lives in the diagonal tile and the fold moved its 3-term sum (LSE +-1 ulp controls and C3n leave it
  bitwise). Short causal rows live in masked tiles, so they keep the accepted p / LSE.
- Numerics reproduction (kda-sm90's checker 9be3622-era numerics.py on its ngate_nfwd3_1 captures, `kda-results/kda-sm90-fwd/nv4g`):
  0 rows beyond the LSE envelope (strict v4: 41, all scale3 / scale10 / outliers); no mean-error failures; zero-input
  O finite (repaired defect).
- A/B vs r17_p1: fwd geo-mean **-12.66%** (job 21178095), **-13.09%** (21178096); no shape slower beyond spread.
- C3n2 (f96a26b, C3n + the non-fold scale floor): -9.02% in one job (21178008); C3n measured -5.8% twice with identical
  code apart from the floor, so treat as unconfirmed.
- An edit error in the first C3g build (prologue descale commented out; O garbage, job 21178005) was caught by the
  numerics run and fixed before any timing was used.

## Status after round 2 of kda-fwd (forward)

- Best: C3g (BM192 three-WG + lazy O rescale + S-descale fold on unmasked steps + RHOPRE + 3 stages + zero-scale floor):
  fwd -12.7% / -13.1% vs the accepted build on the paper shapes, ~1.14 x FA3 fp8 (the accepted build is ~1.31 x).
- What limits it (measured): each consumer WG is a latency-bound serial chain (PV chain -> QK -> softmax); in the 3-WG
  layout the ring turn (PV chain with one rescale wait + promote + QK issue) is ~60% of the step. Every overlap that
  needs more than 160 registers spills (QK split, FA3 order, second temp), extra ring turns or moving QK out of the
  turn are much slower, and O in smem does not fit / is slower.
- The faithful per-64-key P basis costs two ops per score (rho rescale + promote) that no bitwise or round-off
  reformulation removed (PVIL, GRPFOLD, SPLIT2, IL2, GPP variants); kda-fwdport measures the same work at ~24% over FA3
  without scale loads, so FA3 parity under this numerics contract needs latency hiding beyond FA3's own.
- kda-fwdport owns the k / v scale staging (smem) lever (the descale cost is mostly k-scale load latency); it stacks
  on C3g.

### C4 PRETURN (kda-sm90-fwd-c4 b7d70a0 on c1ceece) : READY (bitwise stack, >= 1% in two jobs)

- Change: with 128-key tiles and the lazy O rescale, the lazy rebase, rho_1 and the promote weight depend only on the
  consumed tile's state; they are computed before the ring sync, so the turn holds only the two PV sub-gemms, the
  rescale, the promote and the QK issue. Same values, no flag.
- Outputs: O identical to r17_fwd on all four shapes (screen 21178840). Resources unchanged (D128 STACK 16 / 16 B).
- A/B vs r17_fwd: fwd geo-mean -1.10% (21178841), -1.12% (21178842); no shape slower beyond spread.
- C4b = C4 + KTOK: -0.58% (21178996), -0.75% (21178997): rejected.
- Reference after C3g (job 21178656): FA3 fp8 is 13.9% faster (geo-mean; 11.4% by sum) than r17_fwd on the four shapes
  (r17_fwd ~1.16 x FA3 geo-mean).

## Forward floor: evidence and a proposed corrected floor (Hao: "show the evidence and propose a corrected floor")

- The calibrated sol4 forward floor is the TENSOR floor (tile-granular fp8 FLOPs at the cuBLASLt-measured 1386 TFLOPS).
  The accepted forward (r17_fwd = C3g) is at 59.0% of it and at 1.163 x FA3 fp8 (kda-sm90, 822b827). FA3's own fp8
  forward, with strictly less work on the same shapes and hardware, is therefore at 59.0 x 1.163 = **68.6% of the same
  floor**: the floor assumes the QK / PV gemms run back to back at GEMM efficiency, which no attention forward with a
  dependent softmax between its two gemms reaches here.
- Work the method needs that the floor does not count (measured): the issue floor of the faithful forward is 81% of
  the tensor floor (sol4 counts), versus ~55% for FA3; per 256 keys the faithful P basis adds ~2 ops / score (rho +
  promote) plus the descale / group bookkeeping; kda-fwdport measures exactly this work at +24% over FA3 without scale
  loads (+30% with), and its FA3 replica runs within -3% of FA3, so the harness and building blocks are not the gap.
- Proposed corrected floor: floor_fwd = max(sol4 tensor floor, measured FA3 fp8 forward time on the same shapes)
  (the audit's "FA3 fp8 forward is the practical reference"). Under it the accepted forward is at 1 / 1.163 = 86% of
  the floor; the 90% goal means <= 1.111 x FA3, i.e. ~4.5% more (C4 gives 1.1%; k / v scale staging by kda-fwdport is
  the measured next ~3-7%).

## Final status (kda-fwd, after round 3)

- Accepted: C3g (c1ceece): forward -12.89% / -12.68%, residual-v4 pass on four routes. Before it: C1 (folded into C3g).
- Held for stacking: C4 PRETURN (b7d70a0, bitwise, -1.10% / -1.12%).
- Every lever of the brief has at least two designed variants measured (ledger S1-S51): rho rescale (PVIL, GRPFOLD,
  SPLIT2, IL2, RHOPRE, BN64 / no rho, PRETURN), S descale (DFOLD, fold-on-unmasked, ablation), per-group basis work
  (NOMAX / NOEXP ablations, POLY exp2, BN64), FA3 overlap schemes (BN128 FA3 order, OVL3 / OVL3B, QKSPLIT, QKPRE, QKH,
  OSMEM, OVL1), ping-pong (NOSCHED, ARVEARLY, TT2 / TT3, QKRING, KWEARLY, KTOK), tiles / warpgroups (BN128, BM192,
  BN64, stages 2 / 3 / 4), layout (VCOL), scale loads (VPF; KSPF and smem staging owned by kda-sm90 / kda-fwdport).
- Open: k / v scale staging through smem (kda-fwdport) to stack with C4 toward 3%; FA3 parity needs ~14% more and,
  by the measurements above, latency hiding beyond FA3's own under this numerics contract.

- Integrator decision on the floor proposal (recorded in rule-change-r10.md): not adopted as the goal floor; every
  accepted round reports both the sol4 % and the FA3-referenced % (r17_fwd: 59.0% of sol4, 86% of FA3). Method work
  (per-group basis / Horner / promote) stays counted in the sol4 issue / MUFU floors from the emitted loop. Next step
  unchanged: C4 + kda-fwdport's k / v-scale smem staging, >= 3% combined in two jobs vs r17_fwd.

### Stack C4 + SMSC (kda-fwdport, origin/kda-sm90-fwdport-smsc 80b8151 on b7d70a0) : below 3% (held)

- k / v scales staged into smem with the K TMA (kda-fwdport) on top of C4. Bitwise: O identical to r17_fwd on all four
  shapes (screen s27). A/B vs r17_fwd: fwd geo-mean -2.46% (job 21179751), -2.42% (21179752); job 1 flags B2 32/4
  N8192 e2e fwd+bwd slower beyond spread (its forward is faster). Causal D128 symbol STACK 16 -> 40 B (kda-fwdport
  investigating). Needs ~0.6% more bitwise to clear the 3% stacking bar.

- Tile-temp lever closed (kda-fwdport analysis, agreed): an fp16 temp does not fit FA3 order at 3 WGs (O 64 + S 64 +
  P 16 + T 32 = 176 > 160), needs a ~2^-9 prescale of P (products reach ~1.3e7 > 65504), and its accumulation error
  (~+2e-5 per case) fails residual-v4's 1e-5; a per-WG O slab in smem has S45's LDS / FFMA / STS promote cost.

## DONE (kda-fwd), after Hao's sol4c decision

- Accepted from this helper: C3g (kda-sm90 c1ceece; C1's lazy rescale + descale fold inside it). Held: C4 PRETURN
  (bitwise -1.1%) and the C4 + kda-fwdport SMSC stack (bitwise -2.46% / -2.42%), below the 3% stacking bar.
- Forward status: ~82% of sol4c (sol4-v2 / 0.699), 1.146 x FA3 fp8 on r17_p0s; the 90% sol4c target and FA3 parity are
  not met.
- Why no further lever (per-element instruction count of the steady step, 128-key tile, per thread: 64 scores):
  FFMA (code argument) 64, MUFU (exp2) 64, FMNMX (group max) 64, FADD (row sum over unquantized p, required by the
  denominator contract) 64, F2FP 32: the FA3 core; plus rho rescale 64 + promote 64 (the per-64-key P basis with a
  zero-init tile temp, the fa3fp8_ff precision fix) and ~120 bookkeeping. Removed already: the S-descale FMUL (fold),
  the per-tile alpha FMUL (lazy), the scale-load latency (SMSC), the in-turn rho / weight / rebase work (PRETURN).
  Every remaining candidate measured here either needs more than 160 registers at 3 WGs (spills), adds a branch inside
  the step (spills), moves work across the ring turn (slower), or changes numerics beyond residual-v4 (fp16 temp,
  merged bases). 54 screened variants, at least two per lever (ledger S1-S54).

## Round 18: shared-basis P codes (FWDSHB), directed by kda-sm90 : REJECTED (numerics)

- Design (branches kda-sm90-fwd-shb dbe37f3 = 3 WGs / BN128 on 8bc2425; kda-sm90-fwd-shb2 = 2 WGs / BN256, 2 stages):
  per row and tile, shared basis ms_S = max_g ms_g; group g's codes on ms_S - log2(v_g / v_max) (V-scale ratio folded
  into the codes), so the PV chain is ONE gemm per tile with no rho rescale and one promote; row sum and promote weight
  unchanged in exact arithmetic (sigma_g part_g cancels the offset; weight = v_max 2^(ms_S - mref)).
- Timing vs r17_pk (fwdtime, 4 shapes): 3 WGs **-4.41% / -4.33%** geo (jobs s31 / s32); 2 WGs BN256 **+8.30%** (s32;
  the 3-WG layout matters more than the rho chain).
- residual-v4, default route vs r17_pk (+ LSE envelope, --allow-repair; job nv5, `kda-results/kda-sm90-fwd/nv5`): FAIL, 82 problems.
  Matched O error rises in 10 of 23 cases by 1.3e-3 .. 4.2e-3 (limit 1e-5): gqa3_promote 4.604e-2 -> 5.025e-2,
  min 4.529e-2 -> 4.825e-2, mqa 4.655e-2 -> 4.916e-2, gqa4 4.629e-2 -> 4.846e-2, packed / bhnd / constk ~+1.7e-3,
  heldout +1.3e-3; 7 matched rows beyond the envelope. Cause: the smaller group's codes on the shared basis lose
  precision (more e4m3 subnormals), a real accuracy loss, 100-400 x the tolerance.

## Round 18 (cont.): turn-bound diagnosis and ring-token levers : all REJECTED

- Light phase probe on C3g + C4 + SMSC (diagnostic build, spills 144-160 B so absolute cycles are inflated; job
  lprobe, both 8K and 64K shapes): per WG step ~4020 cycles = pre-turn 370 (9%) + waiting for the ring token 950 (24%)
  + ring turn 1300 (33%: PV gemms, rescale, promote, QK issue) + post-turn 1350 (33%: QK wait + softmax). Three turns
  ~= one step: the forward is RING-TURN-BOUND; the softmax phase has slack.
- TOK2 (two tokens in the named-barrier ring, WG2 skipping its first sync): deadlocks (a WG can be lapped: two
  arrivals land on one named barrier before its sync); job cancelled.
- TOK2b / TOK1s (ring as smem counting semaphores, 2 / 1 tokens; counters in PipelineStorage after a first version
  misaligned the dynamic smem): +28.9% / +48.6% geo vs r17_pk (job s35): the spin loop spills (STACK 136-152 B); the
  second token is worth ~12% against its own 1-token control.
- TOK2p / TOK1p (parity-alternating named barriers, 6 ids: no lapping, no spin; 2 / 1 tokens): +23.1% / +19.8% geo
  (job s36, STACK 72 / 160 B): the extra per-thread ring state spills in the 160-register turn, and the second token no
  longer helps. All bitwise (O identical).
- Reading: at 3 WGs every added per-thread state or branch in the step spills; the turn's content (two PV sub-gemm
  waits, the rescale and the promote, the QK issue) is fixed by the per-group basis and the register budget.

## Round 18 (cont.): one more piece for the C4 + SMSC stack (Hao: stack measured -3.24 / -2.90%)

Stack rebased on r18_c4 as kda-sm90-fwd-stk (30a05d9 + switches commit); four bitwise turn levers built in parallel
(all O identical to r18_c4, resources unchanged), screened 3 reps on the four shapes vs r18_c4 (jobs s37 / s38):

| variant | change | geo vs r18_c4 | stack control in the same job |
|---|---|---:|---:|
| sl1 | group-0 PV sub-gemm issued before the ring sync | -2.51% | -2.45% |
| sl2 | + its retire and the rho rescale before the sync | -2.32% | -2.45% |
| vrel | V stage released after the turn's arrive | -2.55% | -2.30% |
| sl2v | sl2 + vrel | -2.51% | -2.30% |

None adds >= 1% (all within +-0.25% of the stack). Removing the rescale and one wait from the turn (sl2) changes
nothing, so the light probe's "turn-bound" reading was distorted by its own spills. Per-step cost is the same at 8K and
64K (~1.93 us per step per SM, from kernel time / causal step count), so per-tile overheads are not a lever either.

### C4 + SMSC : ACCEPTED as part of S19-fwd-23 (kda-sm90 020b155)

- Stack gated by kda-sm90 as a fresh pair: C4 PRETURN (this helper) + SMSC (kda-fwdport) + the integrator worker's
  S19-fwd-10 (lazy-rebase -inf guard dropped on unmasked steps; next tile's scale reads after the ring-turn arrive),
  PREW2 (next step's rho_1 / promote weight in the previous softmax) and WGID (barrier ids once). Gate 21195830:
  bitwise on all four routes, forward -3.51%; confirmation 21196466 (library 9ffc39ca080e): -3.92%. New accepted root
  `.local/frozen/r19_fwd`.

## Round 19 (on r19_fwd, 1.108 x FA3): softmax-side sensitivity and FLOORPREV

- Timing-only ablations on the r19 source (wrong numerics; 3 reps, 4 shapes, vs r19_fwd; jobs s39 / s40; control
  rebuild r19c +0.05% / -0.14%):

| ablation | removed | geo vs r19_fwd |
|---|---|---:|
| abQ | quad all-reduce of the group maxima (8 SHFL + 8 FMNMX) | **-1.93%** |
| abF | exp2 of the row-sum fold (4 MUFU) | -0.67% |
| abL | lazy-rebase check | -0.09% |
| abS | basis scalar arithmetic (floor, -log2 448) | -0.17% |

  So the softmax is NOT slack: the ~65-cycle quad-shuffle latency of the group-max reduction is exposed on each WG's
  chain (QK wait -> thread max -> quad shuffles -> basis -> exp2). It cannot be hidden bitwise: every group's basis floor
  needs the whole tile's max, i.e. every group's reduction.
- FLOORPREV (kda-sm90-fwd-flp ada5860, named numerics change): floor from the PREVIOUS running max, fused max + softmax
  so group 0's exp2 overlaps group 1's shuffles. +1.03% geo (slower, job s41) and residual-v4 FAIL (job nv6, vs my
  r19 DBG + LSE +-1 ulp controls): scale10 / outliers O error 0.27 -> 0.37 and stale dQ / dK x3. Cause: when the max
  jumps inside a tile the two group bases are no longer within B binades, and the next step's rho
  (v0 / v1) 2^(ms0 - ms1) hits its [1e-30, 1e30] guard. The current-max floor is what bounds |ms0 - ms1| <= B. Rejected.
- My r19 DBG build and LSE +-1 ulp control packages: `kda-work/frozen/r19_dbg`, `r19_dbg_lse{p,m}`.

Raw data (timing JSON, numerics captures, A/B directories) lives outside git under
`~/workspace/haok1402/kda-results/kda-sm90-fwd/` (formerly `kda-work/runs/`); this file keeps the numbers that matter.

### Fallback for residual-v5: accepted forward without FWDDFOLD (kda-sm90-fwd-nofold a8ee6dc, on origin/kda-sm90)

- r19_fwd with `FA3FP11_HD128_FWDDFOLD=0` (BM192 + lazy + C4 PRETURN + SMSC + S19 pieces kept; the no-fold descale path
  already carries the FLT_MIN scale floor). Matched-mode change vs the original kernel: O only (LSE / backward are
  the original's, so the dS row residuals are the original's by construction).
- Timing vs r19_fwd (3 reps, 4 shapes, job s42): **+4.80% geo** (the fold is worth ~4.8% on the current stack);
  i.e. about 1.16 x FA3 fp8 if C3g's fold has to go. Builds: `kda-work/frozen/nofold`, `nofold_dbg`.

## Round 21: residual-v5 / TASK.md error clause: BN256 port of the accepted forward

Directed by kda-sm90 (BM192 is suspect under the relative error clause; the two-tile PV temp does not fit at 3 WGs).
Branches on e7fec9d: `kda-sm90-fwd-bn256` (fold on) and `kda-sm90-fwd-bn256nf` (fold off). 2 WGs x 128 x 256, 2 stages,
FWDLAZY + RHOPRE; the 2-group pieces (PRETURN, PREW2, SMSC) are off at four groups, so ports were added: PRETURN4
(rebase, three rho and the weight before the ring turn) and SMSC for 4 k + 4 v scales (window k8s[4n .. +4)), both bitwise.

Timing vs r19_fwd (3 reps, 4 shapes, job s43): p256 +10.62% geo, q256 (+ PRETURN4 + SMSC4) **+8.94%**, p256nf +15.95%,
q256nf **+13.45%**. REG 168 / STACK 24 (as the original tile), no diagnostics.

Relative error vs the ORIGINAL kernel (frozen/base), per case, 66 finite cases, identical on all 4 routes (job ev1):

| build | O > 1e-5 (worst) | dQ > 1e-5 (worst) | dK > 1e-5 (worst) | dV > 1e-5 (worst) | means (O / dQ / dK / dV) |
|---|---|---|---|---|---|
| r19_fwd (accepted) | 14 (+1.5e-4 outliers) | 11 (+2.3e-3 mqa) | 8 (+3.2e-3 mqa) | 0 (+4.1e-6) | +4.3e-6 / +1.4e-6 / +3.9e-6 / -4.1e-7 |
| q256 | 0 (+1.2e-6) | 1 (+1.0e-5 scale10) | 0 (+6.7e-6) | 0 (+5.0e-6) | -2.0e-8 / +1.3e-6 / -8.3e-7 / -3.1e-7 |
| q256nf | 0 (+1.1e-6) | 0 (+5.3e-6) | 1 (+1.2e-5 outliers) | 1 (+1.0e-5 gqa3) | +2.2e-9 / +6.1e-8 / -3.9e-7 / -3.1e-7 |

O (the forward's own output) passes with both ports (worst +1.2e-6); the dQ / dK / dV movements of every build include
the backward numerics changes already accepted on the head (pass 0 / pass 1), so they are not forward-attributable.

### BN256 port under the residual-v5 row check (v5-abs, c = 1, the decided form) : PASS

- DBG builds from my trees with kda-sm90's r21-v5 DBG extension applied (`kda-work/v5/q256{,nf}`, jit `v5-q256{,nf}-dbg`);
  captures with its `numerics_v5.py` (read-only), checked against the ORIGINAL's DBG rows (its `data/orig`; the
  original's acceptance-suite dump re-captured into my results dir). Job 21204758; raw data in
  `kda-results/kda-sm90-fwd/v5/`.

| suite | rows | q256 (fold on) | q256nf (fold off) | max c needed |
|---|---:|---|---|---:|
| 72 cases x {default, kv, head, streams2} | 334,592 per route | 0 abs-worse (161 ratio-worse) | 0 (72 ratio-worse) | 0.334 |
| stress | 1,703,936 | 0 (78,780 ratio-worse) | 0 (53,580 ratio-worse) | 0.616 |
| acceptance shapes | 11,796,480 | 0 (0) | 0 (0) | 0.012 |

- Both builds include the head's accepted backward changes (EXPBX64, C4), so this is the combination's result.
