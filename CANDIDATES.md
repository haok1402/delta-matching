# kda-pass1 candidates (pass 1, D128 causal matched) -- v2

Helper kda-pass1. Current branch `kda-sm90-pass1-v2` (on `origin/kda-sm90` 3106faf = round 17 with S17-p1-39);
the first campaign is on `kda-sm90-pass1` (switch scaffolding, full history) and `kda-sm90-pass1-c1`.
Levers are compile-time switches, default OFF (the default build is the accepted code). Build hooks: `DM_XDEFS`
(extra -D) and `DM_XFLAGS` (flag-set overrides; defines `KDA_FLAGSET_DEV`, which skips the `dm_flagset.h` check) in
`delta_matching/_buildcfg.py`.

Method: quick screens = `ab.py --grid one --smoke`, 2 alternating reps, B2 N8192 32/4 and 16/4, A = the latest frozen
accepted build (r14_h3 / r17_c2 / r17_p1 as they were accepted; noise ~0.2-0.5%); full A/B = `ab.py --grid paper`,
3 reps, two fresh jobs; equivalence = 72 cases per route vs the frozen r14_h3 dump (outputs of every accepted build
since are bitwise equal to it); numerics = `numerics.py capture` (production x2, DBG x2) checked against the
round-17 certificate captures with my residual-v4 screen (`pass1-aux/p1x/v4check.py`, Hao's rule as stated:
per-case error rise <= 1e-5, per-output mean not rising, strict per-row non-regression on rows whose accepted residual
is <= 1e-3, degenerate set not growing, deterministic). Full screen log: `results/kda_pass1_screens.md` (124 rows).

## READY

### C2: SHFL8, the 8-shuffle dS' transpose (bitwise; >= 1% for stacking)

- Branch `kda-sm90-pass1-c2` (03bfa0f): `KDA_P1_SHFL8` defaults to 1 on top of `kda-sm90-pass1-v2` (all other switches
  off), base = `origin/kda-sm90` 3106faf (S17-p1-39 accepted).
- What it changes: the dS' natural transpose for pass 2. Simulating the H1b byte movement
  (`benchmarks/kda_xpose_sim.py`) shows that every target lane needs WHOLE A-operand words: rdS words 2j + h
  (h = lane >= 16) from lanes srcA and srcB, so the 16 half-useful shuffles become 8 fully useful ones. Phase 0: low lanes
  read srcA, high lanes srcB; phase 1 swapped. A provider picks word 2j or 2j + 1 by (lane bit 2) ^ phase with one
  per-lane-selector PRMT, before the dK issue (rdS is the in-flight A operand afterwards, as in H1b); a receiver
  interleaves its two words with a per-lane PRMT selector. 8 PRMT + 8 SHFL + 8 PRMT instead of 8 + 16 + 8 per step;
  byte-exact for all 32 lanes in the simulation.
- SASS: SHFL.IDX 16 -> 8 per step; REG 168, STACK 8 / 0 (= the accepted build), LOCAL 0, no ptxas diagnostics.
- Bitwise: equivalence vs r14_h3 BITWISE EQUAL on default / kv / head / streams2 (72 cases each); pytest 49 / 3.
- A/B vs frozen r17_p1, `--grid paper`, 3 reps, two jobs (21177436, 21177437; `pass1-aux/runs/full_v2s8_{1,2}`):

| shape | job 1 pass 1 | job 2 pass 1 |
|---|---:|---:|
| B2 H16/4 N8192 | -2.12% | -2.02% |
| B2 H32/4 N8192 | -1.92% | -0.83% |
| B1 H16/4 N65536 | -1.27% | -1.52% |
| B1 H32/4 N65536 | -1.80% | -1.63% |
| geomean | **-1.78%** | **-1.50%** |

  Other kernels within +0.5% (unchanged binaries); e2e fwd+bwd geomean -1.20% / -0.56%.
- With the release flag `FA3FP11_HD128_DOST` 1 -> 2 (the integrator's flag probe 40, bitwise): pass 1 geomean
  **-2.45% / -2.33%** (jobs 21177438 / 21177439, `full_v2s8o_*`).
- Status: **READY (>= 1% bitwise, for stacking)**.

## C1 (TOPFILL=2 + DSDEFER + DVZI + LSEPRE + PV4): SUPERSEDED by the accepted S17-p1-39

C1 was READY on 2026-11-01 (branch `kda-sm90-pass1-c1`, e2add66): pass 1 geomean -4.36% / -4.42% vs r17_c2 on the
paper set (two jobs 21173276 / 21173277), bitwise on all four routes. The integrator's breadth screen found and
accepted the same design independently (RESCALE-EARLY == TOPFILL=2, DSDEFER-SPLIT == DSDEFER; round 17, -4.68% /
-4.87%). Re-measured on top of it (r17_p1), C1's remaining parts add nothing: DVZI +0.18 / -0.39%, LSEPRE +0.80 /
+0.73%, DVZI+LSEPRE -0.19 / +0.10%, PV4 neutral (also the integrator's 25 / 44). **No action needed.**

## Screened on the round-17 base (v2), all bitwise unless noted, pass 1 B2 N8192 32/4, 16/4 vs r17_p1

| variant | what | pass 1 | verdict |
|---|---|---:|---|
| DOST=2 (flag) | Q / dO share the 2-stage pipeline (release flag change; outputs bitwise) | -2.40 / -1.47% | confirms the integrator's flag probe 40 (theirs) |
| FENCE2 | proxy fence every 2nd tile, 4 dS' buffers per WG, pair stores | +3.25 / +3.01% | reject |
| FENCE2 + DOST=2 | | +2.22 / +2.26% | reject |
| PCONVEARLY | P~ e4m3 conversion right after exp (before the dP wait) | -0.61 / -0.01% | neutral (conversions lever, variant 1) |
| PCONVEARLY + DOST=2 | | -1.03 / -1.43% | no better than DOST=2 |
| EXPSPLIT (on r17_c2) | exp rows 0..7 before the dP wait, dS 0..7 interleaved with exp 8..15 | -0.39 / -0.58% | neutral (first version deadlocked: `if constexpr (false)` before a `#pragma unroll` loop lost a pipeline release; fixed with #if) |
| (QST, DOST, DOTST) flag sweep | (2,2,1) / (3,3,1) / (3,2,1) / (3,3,2) / (2,2,2) | -1.02/-1.39, -1.61/-1.36, -0.86/+0.23, +0.07/-0.14, +1.10/+0.75% | nothing beyond DOST=2 (2,2,1); screen noise ~1% (the same config v2o read -2.40/-1.47) |
| SHFL8 | 8-shuffle dS' transpose | -1.67 / -1.86% | **C2** (conversions / packing lever, variant 2) |
| STLATE=1 / =2 | the deferred dS' store issued after the dK issue / after the transpose (not right after the psi barrier) | +1.37 / +1.55%, +0.26 / +0.13% | reject (with SHFL8: -1.01 / -1.26, -1.36 / -1.21 vs SHFL8 -1.67 / -1.86) |

## Output-changing levers vs residual-v4 (default route; all deterministic; pytest 49 / 3)

| candidate | pass 1 screen | worst per-case error rise | mean error per output | well-conditioned rows above accepted + k 2^-23 | verdict |
|---|---:|---:|---|---:|---|
| PW16 (dV promote window 16) | -2.32 / -2.36% | +1.669e-5 (dV, d128_bhnd causal matched + stale) | dV -0.0077%, others 0 | 0 | FAIL (2 cases over 1e-5 by 0.67e-5) |
| PW32 | -4.11 / -3.82% | +3.33e-5 (dV, 8 cases) | dV -0.0041% | 0 | FAIL |
| EXPB (+8 folded into the row bias, pass 0 + pass 1) | -2.44 / -2.23% | +5.2e-7 | ~0 | 22 rows in 6 cases (worst +1.46e-5) | FAIL (any change of P^'s rounding moves per-row residuals) |
| DVZI + LSEPRE + EXPB + PW32 | -6.83 / -5.98% | +3.33e-5 | dV -0.0040% | 22 | FAIL |

Reading: under residual-v4 a change that touches P^ or the dS line cannot pass (the per-row residual is a rounding
residue; moving the roundings moves it up in some rows); a change that only regroups an accumulation (PW) leaves every
residual unchanged and lowers the mean error, but its per-case changes are rounding noise of order 1e-5 and land just
over the 1e-5 tolerance on one shape. Choosing a window alignment that happens to pass would be fitting the test set,
so PW stays rejected.

## Earlier campaign (on r14_h3 / r17_c2), rejected or superseded (details in the screen log)

REDUX 1/2/3 (+3.1 / +1.7 / +0.2%), DVLATE (+3.6%), EARLY 1..5 with and without a liveness kill (+70 .. +200%: wgmma
accumulators in flight across the back-edge pin S / dP, C7511), FOLDSPLIT (+15.9%, spills), XPTOP (+1.4 / +1.8%),
UNROLL2 (+8.9%, spills), TOPFILL=1 / 3 (the promote does not benefit from the S bubble), HALFPSI + P2A / P2B
(pass 1 -2.1 .. -2.9%, pass 2 +10.7 .. +13%), HALFPSI + 2-stage dO/dOt, PP1 / PP2 / PP3 ping-pong (-1.1 .. +1.0%),
POFF producer-warp promote (+4.2%), KRS / VRS (+2.8 / +1.8%, spills; +20% together), DKSPLIT (+3.7%; with KRS +6.5%,
with EARLY still spills), STSEL 1 / 2 (0 / +5.8%), PSIADDR (noise).

## What the measurements say about pass 1's limit (for the 90%-of-floor goal)

- Timing-only ablations (garbage outputs) of the accepted step: removing each vector piece saves nearly its full cost
  (exp -9.5%, dS' transpose + store -11%, dK fold -8.5%, promote -11%, psi -9.3%, conversions -5.3%, ...); all vector
  work -63%; all gemms -30%: vector and tensor work do not overlap; the gemm + TMA skeleton is ~37% of the step.
- Shared-memory bandwidth is not the bound: +16 / +32 KB of LDS per CTA step in the S-wait bubble costs nothing.
- Issue: ~734 executed instructions per warp-step (the static loop has 1026; the promote and rescale arms are 1/8 and
  1/2 of the steps), i.e. ~1470 issue cycles per SMSP per CTA step against a ~3600-cycle step: the vector phases run at
  ~0.65 IPC per SMSP with the two warpgroups' warps interleaving. A lone warp issues slower still, which is why
  ping-pong (one warpgroup's vector work alone under the other's gemms) does not help.
- Every intra-warpgroup overlap design (early S / dP issue, K / V held in registers, split dK) is blocked by the
  register budget: dK 64 + dV 64 + dK temp 64 + S / dP 64 of 240.
- So a ~2x pass 1 needs a different dataflow (fewer live accumulators per warp, more warps per SMSP, or less vector work
  per FLOP), i.e. the structural redesign lane (kda-p1redesign / kda-fuse), not schedule tweaks on P1-R.

## Final summary (2026-11-02, helper kda-pass1)

Deliverables:
- **C1** (TOPFILL=2 + DSDEFER + DVZI + LSEPRE + PV4, bitwise, -4.36% / -4.42% geomean, two jobs): its two effective parts
  were found and accepted independently as S17-p1-39 (round 17); the rest is neutral on top of it. Superseded.
- **C2** (SHFL8, bitwise, -1.78% / -1.50% geomean, two jobs, 4 routes bitwise): READY for stacking; with DOST 1 -> 2
  -2.45% / -2.33%. Branch `kda-sm90-pass1-c2`.

Lever coverage (each with >= 2 designed variants, ~130 screened builds in `results/kda_pass1_screens.md`):

| lever | variants | outcome |
|---|---|---|
| instruction cuts: scaling | LSEPRE, EXPB, dPsum/LSE moves (integrator 29 / 35 / 41) | LSEPRE neutral on r17_p1; EXPB -2.4% but fails residual-v4 |
| exp2 setup | EXPB, EXPSPLIT | EXPSPLIT neutral; EXPB as above |
| conversions / packing | PCONVEARLY, SHFL8 | **SHFL8 -1.6% (C2)**; PCONVEARLY neutral |
| psi amax | REDUX 1 / 2 / 3 (and the integrator's AMAXTREE) | +0.2 .. +3.1%, none helps |
| dK fold | FOLDSPLIT, DKSPLIT | spills / exposed half gemm |
| dV promote | TOPFILL=1 / 3, PV4, POFF, PW16 / 32 | bitwise ones neutral or slower; PW fails residual-v4 by ~0.7e-5 on one shape |
| per-warpgroup psi (decoupling) | HALFPSI + P2A, + P2B, + 2-stage dO | pass 1 -2..-3%, pass 2 +11..13%; rejected (integrator: pass1+pass2 +0.1..+0.8%) |
| ping-pong over all four gemms | PP1 (offset), PP2 (S/dP turns), PP3 (both batches), +/- 2-stage dO, +/- EARLY | -1.1 .. +1.0%; no gain |
| structural (in-kernel) | EARLY 1..5 (+ liveness kill), KRS / VRS, DKSPLIT (+EARLY / +KRS), POFF, UNROLL2, STSEL | all limited by the 240-register budget or slower |
| pipeline depth | DOST / DOTST / QST sweeps | only DOST=2 (the integrator's finding) helps |
| store / fence schedule | DSDEFER (== S17), FENCE2, STLATE 1 / 2, XPTOP | DSDEFER accepted (S17); others slower |

Where pass 1 stands against the sol4-v2 floor (tensor-controlled, 78.6 ms on the paper set): 188.5 ms (41.7%) on r16;
~179 ms (~43.8%) after S17-p1-39; with C2 ~176 ms (~44.5%), with DOST=2 as well ~45%. The measurements above
(ablations: vector and tensor work fully serialized; vector phases at ~0.65 IPC per SMSP; smem bandwidth not binding;
every overlap design blocked by dK 64 + dV 64 + dK-temp 64 + S/dP 64 registers of 240) say a 2x on P1-R needs a
dataflow with fewer live accumulators per warp or more warps per SMSP: the structural lane (kda-p1redesign /
kda-fuse), not further schedule tweaks of this kernel. Numerics: under residual-v4, any lever that changes P^ or the dS
line fails the per-row test by construction (rounding residues move); only accumulation-regrouping levers (PW) can pass,
and PW16 misses the 1e-5 per-case error tolerance by 0.67e-5 on one shape while lowering the mean.

### Addendum: the integrator's residual-v4 checker (9be3622: LSE +-1 ulp envelope, 1e-5 per matched case)

Same references as the integrator's gates (`.local/rounds/r17/ngate_nfwd3n_2`, `results/r17/cert/defects_default.json`):
EXPB FAIL (fresh captures, job 21177918: 8 well-conditioned rows above accepted + round-off in gqa4 / gqa3_promote /
bhnd / heldout; 22 without the envelope); PW16 FAIL (d128_bhnd matched dV +1.67e-5); PW32 FAIL (also gqa3_promote dV
+3.3e-5). 9be3622 has no csrc change vs 3106faf: C2 stands as measured.

### Addendum: C2 stacks on r17_fwd (kda-sm90 c1ceece, forward C3g accepted), paper grid, two jobs each

| stack | pass 1 geomean job 1 / job 2 | bitwise vs r17_fwd |
|---|---:|---|
| SHFL8 | -2.20% / -1.74% | (C2) |
| SHFL8 + DOST=2 | -1.95% / -2.20% | 4 routes |
| SHFL8 + QST=3, DOST=3 | -2.15% / -2.45% | default route |

None reaches the 3% bar; C2 stays held for stacking with another bitwise piece.

### Addendum: on 7a06674 (r17_pk, EXPBX64 accepted), paper grid, two jobs each

| build | pass 1 geomean job 1 / job 2 |
|---|---:|
| SHFL8 + QST 3 / DOST 3 (`kda-sm90-pass1-c3pk`) | -1.18% / -0.50% |
| SHFL8 alone (`kda-sm90-pass1-c2pk`) | **-1.10% / -1.07%** |
| QST 3 / DOST 3 alone | +0.25% / +0.23% |

The integrator's full gate of the c3 stack on r17_p0s measured -2.26% (job 21181578). On r17_pk the flags no longer
help, so the stack piece is **C2 on r17_pk: `kda-sm90-pass1-c2pk`**, READY for stacking (>= 1% bitwise in two jobs;
BITWISE EQUAL vs r17_pk on all four routes, job 21182433; REG 168, STACK 0, no diagnostics).

## DONE: per-element instruction ledger on r17_pk (+ SHFL8), 2026-11-02

Unmasked step loop, causal: 993 static / ~698 executed instructions per warp per m-block step (32 elements per
thread). Per element: exp line 3 (FFMA s * dsc - b, F1 clamp FMNMX, MUFU.EX2; EXPBX64 already folded the 2^8 offset
into a per-row FFMA bias), dS line 2 (FFMA, FMUL), psi amax 1 (FMNMX), dS' scale 1 (FMUL), e4m3 conversions 1 (P~ and
dS' F2FP pairs) = 8 / element = 256 per step; per row 32 (bias, dPsum * k8s); dK fold 64 FFMA; dV rescale 64 FMUL on
half the steps; dV promote ~33 averaged. What is left and why it stays:
- clamp FMNMX: the F1 cliff guard (fixed by the numerics contract); a result-side clamp costs the same.
- amax FMNMX / scale FMUL / F2FP: one op per element each; sm90 has no 3-input FMNMX and no scaled e4m3 convert;
  packed (bf16x2 / f16x2) variants need a conversion per pair and break even.
- dS line FFMA + FMUL: minimal for P * (dP * a - b).
- dK fold (2 / element): needed by the plan-A per-tile fp32 fold; accumulating dK across tiles in the wgmma accumulator
  (any window) changes the accumulation and, like PW16 / PW32 on dV, moves per-case dK errors by more than 1e-5;
  power-of-two psi with paired folds changes the dS' codes (per-case dK / dQ errors at quantization level).
- dV promote / rescale: bitwise variants measured neutral (TOPFILL 1 / 3, PV4, POFF, DVZI); PW fails residual-v4.
- per-row work and overhead: LSEPRE, STSEL, PSIADDR, DVZI measured neutral.
The last instruction cut that paid was SHFL8 (-8 SHFL per warp-step, ~-1.1% on r17_pk, consistent with the
~0.05-point-per-instruction sensitivity). I have no further grounded per-element cut: **kda-pass1 is DONE**;
the held piece is `kda-sm90-pass1-c2pk` (SHFL8, -1.10% / -1.07%, bitwise, four routes).

## C4: SHFL8 + NOCLAMP=4 (F1 clamp only on the masked-tile line), pass 1 + pass 0; named numerics change, bitwise on the suite (2026-11-02)

- Branch `kda-sm90-pass1-c4` (56c3510) on 7a06674 (r17_pk). Default build = the measured build (pass-1 and pass-0 SASS
  identical); REG 168, STACK 0, no diagnostics.
- The F1 clamp min(t, 12) stays on the certified line of the tiles pass 0 masks (EXPBX64's predicate: same tiles in
  both passes, so P^ stays identical) and is dropped on the folded line elsewhere: -32 FMNMX per thread per step.
- Screens (vs SHFL8 on r17_pk): NOCLAMP=1 (clamp off everywhere) pass 1 -2.1 / -2.3%, pass 0 -4.0 / -4.3%, but NaN
  dK / dV on the all-zero defect case (masked -inf x underflowed descale = NaN, which the clamp maps to 12): fails the
  numerics gate (route default, 29 problems, all d128_zeros). NOCLAMP=2 (clamp only if the step's LSE / descale is
  non-finite): +7.6 / +7.1% slower (loop duplication) and still NaN. NOCLAMP=4: pass 1 -3.22 / -2.24%, pass 0 -4.44 /
  -3.87%, BITWISE EQUAL.
- Equivalence vs r17_pk: BITWISE EQUAL on default / kv / head / streams2 (72 cases each); pytest 49 / 3.
- Paper set vs r17_pk, 3 reps, two jobs (21191218 / 21191219): pass 1 geomean **-2.93% / -2.93%**, pass 0 **-3.39% /
  -3.21%**, e2e fwd+bwd -1.80% / -1.90%.
- Status: **ACCEPTED** as kda-sm90 65d4424 (gate 21191514: pass 0 -3.53%, pass 1 -3.31%, e2e -2.21%; confirmation
  21191853: pass 0 -3.86%, pass 1 -3.08%, e2e -2.31%; bitwise on four routes); new root `r18_c4`.
- Pass 1 reusing pass 0's S / dP or computing the matched delta itself: already prototyped and priced by kda-fuse (F1a
  PDL two-phase +0.11%; F1b delta in pass 1: row-sum work +17% of pass 1, infeasible at 64K; F2a-c handoff costs more
  HBM time than pass 1), not duplicated here.


Status after C4 (2026-11-02): kda-pass1 is DONE again. No further grounded per-element cut in pass 1 on r18_c4
(ledger above; the dK fold and pass-0/1 reuse are covered by kda-p1split / kda-p1redesign / kda-fuse).

## Round on r18_c4 (2026-11-02, Hao: contract-fixed instructions, the dK fold, pass-0 reuse)

| variant | what | pass 1 screen vs r18_c4 (32/4, 16/4) | numerics | verdict |
|---|---|---:|---|---|
| GUARDLITE=1 | drop the F2b psi floor + its running g_max | +0.03 / +0.37% | bitwise (72 cases) | no gain |
| GUARDLITE=2 | drop the dV window ratio clamp | -0.60 / -0.20% | bitwise | no gain (noise) |
| GUARDLITE=3 | both | -0.40 / +0.45% | bitwise | no gain |

- The remaining per-element instructions after C4 (exp FFMA + MUFU, dS FFMA + FMUL, amax FMNMX, scale FMUL, e4m3
  conversions, dK fold) are not guards: changing any of them changes P^ (moves per-row residuals: fails residual-v4's
  per-row test, as EXPB did), the dS' codes (per-case dK / dQ errors at quantization level, far above the 1e-5 per-case
  tolerance) or the dK accumulation (in-wgmma dK, like PW on dV, measured +1.7e-5 .. +3.3e-5 per case on dV alone).
- dK fold: measured elsewhere, not repeated: RIP in-place dK (kda-p1redesign) +3.78% slower; deleting the fold is a -8%
  timing bound (kda-p1split) that no design reaching residual-v4 approaches.
- Pass-0 reuse: kda-fuse priced every handoff (F2a-c) and built F1a / F1b; per-tile statistics from pass 0 can only bound
  psi (a different dS' code set: fails residual-v4), so no variant was worth a build.
Status: DONE on r18_c4.
