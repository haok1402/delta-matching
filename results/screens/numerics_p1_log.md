# Round 18, worker S18-nm: named numerics changes in pass 1's per-element path (F1 exp clamp, fp32 dK fold)

Screen: `screen.sh` (2-rep A/B, B2 N8192 H16/4 and H32/4, causal matched). The A reference is r17_pk for 01-11 and 12, and
r18_c4 (C4 accepted, kda-sm90 65d4424) for 12c/12d. Percentages are B vs A; negative = faster. Screen noise is about +-0.5..1%.
"bitwise" = `equivalence.py compare` on the default route (72 cases) against r17_pk's dump.
Each tree is `.local/wt/S18-nm-NN`, the library is `.local/jit/S18-nm-NN/_C.so`, and the patch is `.local/screens/S18-nm-NN/variant.patch`.
01-09 are diffed against a6cf7cf csrc and 10-12 against origin/kda-sm90-pass1-c4 csrc (identical to HEAD 65d4424 csrc).

## Exp clamp (F1). Superseded mid-task by kda-pass1's C4, which the integrator accepted

| tag | change | pass1 16/4, 32/4 | pass0 | e2e | default route | REG/STACK (P1R) | verdict |
|---|---|---:|---:|---:|---|---|---|
| ex2 check | exhaustive GPU test over all 2^32 fp32 inputs of `ex2.approx.ftz`: `ex2(min(y,12)) == min(ex2(y),4096)` bitwise, 0 mismatches; y>12 implies ex2(y) >= 464, 0 exceptions | - | - | - | - | - | proof tool (job 21190811, `.local/rounds/r18-nm/ex2check.cu`) |
| S18-nm-01 | pass 1 only: F1 clamp dropped on the unmasked (EXPBX) loop | -1.09 / -1.14 (rep: -1.23 / -1.09) | ~0 | ~0 | BITWISE EQUAL (the clamp never binds on the suite) | 168/0 | the clamp's whole value is ~1.1%. Bound not provable inside pass 1 |
| S18-nm-02 | CLAMPDET: unclamped exp, P~ e4m3 codes converted early, a warp tests for any saturated code (byte+2 trick, 8 IADD + ORs + vote) and repairs with min(P^,4096) only then. Exact by the ex2 identity | +5.58 / +4.37 | ~0 | +1.0 / +0.85 | BITWISE EQUAL | 168/8 | reject (the branch and the early convert break the schedule) |
| S18-nm-03 | 02 + SHFL8 | - | - | - | - | - | cancelled (02 slower). The tree also lacks SHFL8 (git apply skipped it silently) |
| S18-nm-04 | meant to be 01 + SHFL8. SHFL8 was NOT applied (silent git apply skip), so it is effectively 01 | -1.07 / -1.90 (rep -1.14 / -1.26) | ~0 | - | BITWISE EQUAL | 168/0 | invalid as a stack. Counts as a third 01 sample |
| S18-nm-06 | 01 + SHFL8 (correct stack) | **-3.51 / -2.60 (rep -3.40 / -3.71)** | ~0 | -0.9 / -7.6 (rep -1.7 / -1.1) | BITWISE EQUAL | 168/0 | >= 3% in 2 screens, but the bound is unproven. This is the same idea as C4 (accepted with the clamp kept on the masked-tile line) |
| S18-nm-07 | SHFL8 alone (reference) | -1.20 / -0.63 (rep -1.06 / -1.76) | ~0 | - | (bitwise per kda-pass1) | 168/0 | reference |
| S18-nm-09 | P0FLAG, the exact version: pass 0 drops its clamp on non-masked tiles, tracks a NaN-propagating max of y (same op count), repairs per warp via the ex2 identity, and writes per-(m-block, warp) flag words (y>11 or NaN) behind the delta in the same storage (host registry of delta_fp8 storages). Pass 1 picks a clamp-free step when the m-block's flags are all 0. Bitwise by construction | +12.38 / +11.98 | +5.62 / +5.19 | +6.0 / +5.3 | BITWISE EQUAL | 168/48 | reject: two step copies spill, and pass 0's vote adds 5% |
| S18-nm-08 | 09 + SHFL8 | +16.02 / +16.19 | +5.56 / +5.97 | +7.8 / +7.4 | BITWISE EQUAL | 168/48 | reject |

Clamp reading: the clamp is worth ~1.1% of pass 1 alone and ~2% more when stacked with SHFL8 (06: about -3.3% total vs SHFL8's
~-1.2%). Every exact in-kernel form I built costs more than it saves: a per-tile detect plus repair (02), or a pass-0-precomputed flag with two
step bodies (08/09). C4 (masked-line-only clamp, accepted) takes this lever. Per the coordinator I stopped clamp work there.

## fp32 dK fold (on C4 = HEAD 65d4424)

| tag | change | pass1 16/4, 32/4 | pass0 | e2e | default route residual-v4 | REG/STACK | verdict |
|---|---|---:|---:|---:|---|---|---|
| S18-nm-05 | timing probe (wrong dK, on a6cf7cf): deferred fold every 2 steps, the dK temp kept live across the next S/dP phase | +29.5 / +28.8 | 0 | +11.4 / +11.7 | (probe) | 168/**208** | the deferred-fold structure spills (dK 64 + dV 64 + temp 64 + S/dP 64 > 240): dead |
| S18-nm-11 | C4 tree as is (reference vs r17_pk) | -3.50 / -2.62 | -3.90 / -3.16 | -2.9 / -3.6 | (accepted C4) | 168/0 | reference |
| S18-nm-10 | DKACC: no per-tile fp32 temp and fold. dK accumulates in the wgmma accumulator in a running basis G = max g. Tiles with g <= G are coded with psi' = 256 G / q8sc (coarser dS' codes, published to pass 2). A growing basis rescales the accumulator (rare) | -1.95 / -2.29 vs r17_pk (C4 alone -3.50 / -2.62) | -3.8 / -4.2 | - | **FAIL, 34 problems**: no per-row residual regressions (P^ unchanged), but dQ/dK error rises of 2e-5..1.2e-3 on every matched case (e.g. mqa dQ +3.8e-4, outliers dQ +1.2e-3, scale10 +5e-4); stale-full outliers dK +1.07e-3 > 1e-3; mean dQ/dK rise. Job 21191902, `.local/rounds/r18-nm/num_S18-nm-10/` | 168/8 | reject (numerics), and slower than C4 |
| S18-nm-12 | DKACC without divisions on the common path (psi_inv = q8sc / (256 G)). Numerics are the same as 10 up to an ulp in the published psi | -3.86 / -3.60 vs r17_pk. **vs r18_c4: -0.43 / +0.09, rep -0.41 / +0.14** | ~0 vs c4 | -0.2/-0.8, -2.3/-0.3 | fails like 10 (same codes/accumulation design) | 168/8 | reject: neutral vs C4 AND fails residual-v4 |

dK fold reading: removing all 64 fold FFMA per warp-step (and the 64-register temp) gains nothing measurable on top of C4
(12c/12d: within noise). The step waits for the dK gemm retire either way, so the fold is not the binding cost once the clamp is gone.
Forms that keep the per-tile codes either need the temp live across steps (spills, 05) or regroup the accumulation (DKACC).
DKACC moves dQ/dK errors by 10-100x the 1e-5 tolerance. bf16x2 / packed accumulation would be strictly coarser than DKACC.
sm90 has no packed fp32 FMA (FFMA2 is sm100+), so there is no cheaper exact fold.

## Verdict

No new candidate from this worker. The only >= 3% result (06, NOCLAMP-unmasked + SHFL8, bitwise on the suite) is the
lever the accepted C4 already ships in a better form. Its exact variants (02, 08/09) are slower. The dK fold lever is neutral in time
on C4 (12: -0.4 / +0.1%) and fails residual-v4 on dQ/dK error rises (10). Numerics checks used r17_pk's references (r18_c4 DBG /
LSE-control roots did not exist yet; C4 is bitwise on all certification cases).
Scripts: `.local/rounds/r18-nm/ncheck.sh` (numerics screen), `p0flag_edit.py`, `ex2check.cu`. Reference captures: `.local/rounds/r18-nm/refs/`.
