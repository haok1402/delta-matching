# S18-st: structural redesign of the D128 matched backward (pass 1 reusing pass 0's work)

Worker S18-st, round 18. Base: kda-sm90 a6cf7cf (accepted build r17_pk). All A/Bs are vs `.local/frozen/r17_pk`.
They were taken before the integrator accepted C4 (65d4424, r18_c4).
Patches: `.local/screens/S18-st-NN/variant.patch` (diff -ruN against a pristine `git archive a6cf7cf`, kept in
`.local/rounds/r18-st/pristine`). Trees: `.local/wt/S18-st-NN`. Libraries: `.local/jit/S18-st-NN/_C.so`.
Summary tool: `.local/rounds/r18-st/tools/summ.py`.

Metrics, all from 2-rep smoke A/Bs (`screen.sh`):
- "bwd" = the summed kernel time of pass0 + pass1 + pass2 + glue + prod_bwd + twin.
- e2e = fwd+bwd end to end.
- Shapes: B2 N8192 at H16/4 and H32/4. st-07 and st-08 also ran B1 N65536.

## Verdict

**Neither family pays. There is no candidate.**

- Family B: even the most optimistic prototype loses on the whole backward.
  - It is infeasible at N65536.
  - Its per-step cross-CTA rendezvous alone doubles pass 1.
- Family A: the bytes kill it.
  - Handing products over through HBM costs about 2x the whole backward.
  - Even an ideal free on-chip handoff would only save 8-9% of pass 1.
- Per-tile statistics (A-stat): the only part that paid was skipping the exp clamp, and C4 has since taken that lever.
  My own bitwise-by-proof versions came out slower.

## What is new relative to kda-fuse (origin/kda-sm90-fuse CANDIDATES.md)

kda-fuse already covered these, and I build on their numbers rather than re-measuring them:
- F1a: PDL two-phase. Gain +0.11%.
- F1b: row-sum arithmetic. +17.3% on pass 1, measured on r17_p1; infeasible at 64K.
- F2a-c: storage priced analytically.

What this worker adds:
1. **B0:** a measured whole-backward ceiling for "pass 0 disappears" on r17_pk.
2. **B1:** a cheaper realization of the in-pass-1 row sums, on r17_pk.
3. **B2:** the first measured lower bound of the cross-CTA rendezvous that F1b only priced in words.
4. **A0:** the first measured ceiling of the compute side of family A (pass 1 with S, dP and exp handed over for free on chip).
   It shows the handoff could save little even if the bytes were free.
5. **A-stat:** the clamp as a per-row statistic (st-06/07/08), with a proof that the clamp is the identity for rows with
   finite |LSE| < 4096.
6. **Analytic, new:** a bidirectional lockstep schedule that would make the 8K wavefront load-balanced (below).

## Family B: pass 1 computes the matched delta itself; pass 0 disappears

Design constraint:
- delta(m) needs the partials of every key block n <= m/2 before any dS of row m can be formed.
- So pass 1 either needs all contributors of a row co-resident and in lockstep, or it must store or recompute.
  Recomputing is pass 0.

New schedule idea (analytic):
- Two heads per 64-SM group run in opposite directions.
  - Head A's key block j processes rows in descending order. Every contributor of a row is then naturally at that row
    at the same step: a decreasing triangle.
  - Head B's key block 63-j processes rows in ascending order, skewed: an increasing triangle.
- Together the two triangles fill a 64 x 130-step rectangle with no idle steps.
- At 8K this uses 128 of 132 SMs (~3% loss).
- At 64K a row needs 512 co-resident key-block CTAs, more than 132 SMs, so it is impossible there.
- A partial-band hybrid (pass 0 keeps the far keys) still requires the same lockstep. It frees at most ~44% of pass 0
  at 64K and changes the delta fold (numerics).

Bitwise is infeasible:
- Pass 0's fmaf chain per (row, 32-key half, quad lane) runs over keys {2t, 2t+1, 2t+8, ...}.
- In pass 1's swapAB layout those keys sit on 2 lanes x 2 warps, so the chain would need a cross-lane, cross-warp
  serial hop per element.
- A non-bitwise delta must pass residual-v4 per-row non-regression. kda-pass0's DEFER, LAZY and FOLD all failed that.

| tag | description | bitwise / numerics | pass0 | pass1 | pass2 | bwd (16/4, 32/4) | e2e fwd+bwd | REG / STACK (Fp8R causal) | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| S18-st-01 | PROBE B0: host removes pass 0 (one 1x1x256 dummy launch); delta = 0 | PROBE (numerics wrong) | -98.8 / -99.4% | +0.3 / -0.9% | 0 / 0% | -17.6 / -18.7% | -13.2 / -17.3% | 168 / 0 (r17_pk lib) | ceiling of any B |
| S18-st-02 | PROBE B1: B0 + pass 1 forms per-(row, WG) partials in its own order (FMUL+FFMA per key pair; 32 slots via parity smem written before the psi barrier; 16-slot sums; L2 sink). No rendezvous | PROBE | -98.8 / -99.4% | **+14.4 / +12.6%** | 0 / 0% | -9.2 / -10.6% | -5.9 / -7.0% | 168 / 8 | optimistic ceiling only |
| S18-st-05 | PROBE B2: B1 + the LOWER BOUND of the per-step rendezvous before the dS line (WG smem reduce, L2 publish, fence, atomic, acquire re-read, 8 partials read back, delta via smem into the dS line). Never waits for other CTAs | PROBE | -98.8 / -99.4% | **+101.5 / +99.2%** | 0 / 0% | **+42.2 / +42.0%** | +29.9 / +28.3% | 168 / 8 | **B REJECTED** |

Reading of the B probes:
- B1 already costs +13-14% of pass 1 for the arithmetic alone. kda-fuse's version cost +17%.
- The rendezvous chain is about one full step (~1.7 us) per step.
- Lag-1 pipelining to hide it is not viable:
  - It needs a 64 KB fp32 P/dP buffer per CTA. Pass 1 has ~68 KB smem free, and B1's reduction buffer also has to fit.
  - The 240-register file is full. kda-p1split found register-freed S/dP pipelining 3.5-4% slower.
- Even if the chain were hidden completely, the best case is about -9% bwd at 8K and 0 at 64K (infeasible). That is a
  geomean of about -4.6%.
- From that come the costs of: the lag buffer traffic (~+5% pass 1), lockstep skew, 128/132 SMs (+3%), and a numerics
  change that is likely to fail residual-v4.
- Net is at or above 0. Two designed variants were measured (B1, B2), so the lever counts as failed.

## Family A: pass 0 stores what pass 1 needs

A bitwise handoff needs fp32 P and fp32 dP, 8 B per visited element. The dS line uses the fp32 exp output, not P^.

Measured transport floor (calibrated 3026 GB/s; write in pass 0 plus read in pass 1):

| shape | elements | traffic | floor | whole backward today |
|---|---:|---:|---:|---:|
| 8K 16/4 | 1.07 G | 17.2 GB | 5.7 ms | 2.9 ms |
| 64K 16/4 | — | 550 GB | 182 ms | 88 ms |

- The non-bitwise e4m3 form (F2c) is also out: its dS precision would fail residual-v4.
- Diagonal-only handoff (pass 1's two masked m-blocks per CTA):
  - Traffic at 8K 16/4 is 268 MB written plus read, 0.18 ms, which is 6% of the backward.
  - The saving is at most A0's -8% on ~2/65 of the steps, about 0.25% of pass 1.

| tag | description | bitwise / numerics | pass0 | pass1 | pass2 | bwd | e2e | REG / STACK | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| S18-st-03 | PROBE A0: pass 1 issues no S / dP gemm, no mask, no exp; 32 P + 32 dP per thread from 16 LDS.128 of on-chip smem (the ideal handoff, zero DRAM) | PROBE | 0 / -1% | **-7.9 / -9.0%** | -5.8 / -5.4% (data artifact: garbage dS' → zero tiles) | -5.5 / -6.4% | -6.9 / -9.4% | 168 / 0 (non-causal: C7512) | ceiling only |
| S18-st-04 | PROBE A0-diag: A0 on the masked steps only | PROBE | 0 | +25.3 / +25.3% | -0.7 / -1.1% | +14.8 / +15.2% | +11.7 / +12.8% | 168 / 136 (spill) | invalid (spill); analytic bound 0.25% of pass 1 |

**A REJECTED.**
- The ideal (free) handoff saves only 8-9% of pass 1, about 5% of the backward.
- The real transport floor is about +190% of the backward. That is ~30x the possible saving.

## A-stat: a per-row statistic lets passes 0 and 1 skip the F1 clamp (bitwise by proof)

The proof:
- The forward's row_sum includes the row max's own term (~1). FWDLAZY leaves row_max, row_sum and LSE exact.
- Therefore LSE >= max_j S_j up to round-off.
- So y = S·dsc − (LSE·log2e − 8) ≤ 8 + ε with ε < 1, for any row with finite |LSE| < 4096. ε stays below 1 even under
  the 14-bit fp8-accumulation pessimism.
- Rows with NaN, inf or huge LSE keep the clamp.

This gives a bitwise justification for dropping the clamp on unmasked tiles, which C4 does unconditionally.

| tag | description | bitwise | pass0 | pass1 | bwd | e2e | REG / STACK | verdict |
|---|---|---|---:|---:|---:|---:|---|---|
| S18-st-06 | PROBE: clamp dropped on every unmasked tile in pass 0 and pass 1, no flag test (ceiling) | BITWISE EQUAL, 72 cases (default route) | -4.2 / -4.7% | -1.6 / -2.0% | -1.7 / -2.1% | -0.9 / +2.3% | 168 / 0, 96 / 0 | ceiling < 3% bwd |
| S18-st-07 | P0CLF per-warp flag (`__all_sync` on the warp's rows) + P1CLF per-step per-warp LSE check, runtime branch around the exp loop | BITWISE EQUAL, 72 cases | +1.5 / +1.6 / +0.6 / +0.7% | +7.6 / +7.6 / +5.7 / +6.0% | +4.8 / +4.9 / +3.7 / +4.0% (8K, 8K, 64K, 64K) | +2.5 / +4.3 / +2.5 / +3.0% | 168 / 0 | rejected (the branch breaks scheduling) |
| S18-st-08 | per-CTA flag in pass 0 (bar.red.and, the whole mask-free loop duplicated); pass 1 next-step flag via warp-0 LDG + vote, published before the psi barrier, two compile-time step instantiations | BITWISE EQUAL, 72 cases | -2.0 / -2.1 / -4.7 / -2.9% | +13.2 / +12.3 / +10.5 / +10.3% | +7.5 / +7.1 / +5.9 / +6.1% | +5.2 / +6.6 / +4.1 / +4.3% | Fp8R 168 / 56 (spill), DeltaV6 W4 96 / 16 | rejected (spills) |

Lever status:
- **Superseded:** C4 (65d4424) drops the clamp on unmasked tiles unconditionally: pass 0 -3.9%, pass 1 -3.1%,
  bitwise on the suite.
- **Hand-off:** the |LSE| < 4096 argument above is a ready proof that C4's drop is the identity outside degenerate LSE.
  If a reviewer wants a guard, a per-CTA check is the only cheap form, but st-08 shows that duplicating code costs more
  than the clamp in pass 1.

## Candidates

None reaches a whole-backward gain of 3% or more. Nothing goes to the gate.
