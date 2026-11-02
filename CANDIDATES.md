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
