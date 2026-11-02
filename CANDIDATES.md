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

## Candidates

### C1 FWDLD + RHOPRE (commit e0db18e on kda-sm90-fwd; build `kda-work/frozen/c1c`) : READY (timing; numerics certification by kda-sm90)

- What it changes (D128 PV-first forward only; d256 untouched; new certified flag `FA3FP11_HD128_FWDLD=1`):
  1. S descale folded into the code argument (group maxima on raw codes and descaled once per row / group: gm, ms,
     row_max, alpha bitwise; argument fma(x, RN(q8sc k8sc l2), -ms) instead of fma(RN(x q8sc k8sc), l2, -ms));
  2. lazy O rescale: tOrO kept in a lagging per-row reference (rebased, warp-uniform, before the promote, only when the
     running max drifts > 2^8 from it); the per-tile tOrO *= alpha is gone; finalize folds the reference into 1/l;
  3. RHOPRE: the three rho of the PV chain computed before it (bitwise on its own).
- Outputs: NOT bitwise (named numerics change, round-off class). O changes (lazy + fold); LSE changes at round-off (the
  row sum takes the re-rounded p of the fold), hence dQ / dK / dV change at round-off too. Per-row evidence
  (`kda-work/runs/n2`, fp32 reference, TF32 off, job 21172655): O relative L2 vs fp32 equal to A within 6 digits on
  all six cases (randn x1 / x3 / x10 / x30 logit scale, B1 H16/4 N8192, B2 H4 N2048, B1 H4/1 N16384); per-row error
  differences symmetric (e.g. B1 H16/4 N8192: 39 of 131072 rows worse by > 1e-4 abs, max +1.07e-3, min -1.95e-3 =
  e4m3 code flips at rounding boundaries); no NaN / Inf (the first lazy version gave NaN at x10 / x30; fixed).
- Resources: REG 168, STACK / LOCAL unchanged on every forward symbol (D128 24 / 24, LOCAL 0); no spill / C7510 /
  serialization diagnostics. Steady-state loop 1288 -> 1191 static instructions (FMUL 411 -> 295 static incl. the rare
  rebase branch).
- A/B (benchmarks/ab.py run --grid paper --target fwd, A = r14_h3, 3 reps, two fresh jobs):

| job | fwd geo-mean | fwd sum ms | e2e fwd+bwd | shapes slower beyond spread | decision |
|---|---:|---:|---:|---|---|
| 21172873 (`runs/ab_c1c_1`) | **-7.05%** | 75.682 -> 70.270 (-7.15%) | -0.92% | none | ACCEPT (this job) |
| 21172874 (`runs/ab_c1c_2`) | **-6.79%** | 75.229 -> 70.060 (-6.87%) | -1.65% | none | ACCEPT (this job) |

- Still needed for acceptance (kda-sm90): `gate.py --mode numerics` (four routes, DBG pairs) under Hao's residual
  contract; pytest on the build.
