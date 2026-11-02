# kda-p1packed candidates (pass 1 at head dim 128: per-element arithmetic)

Helper kda-p1packed, branch `kda-sm90-p1packed` (on `origin/kda-sm90` 9be3622 = r17_p1 accepted + residual-v4 decisions).
Lever: the per-element arithmetic of the pass-1 step (packed two-wide math, ex2 f16x2, scaling fusion, e4m3x2
conversions from packed registers, per-element conversions between the S / P / dP / dS stages, hand PTX).
Every lever is a compile-time switch, default OFF (`KDA_PK_*` in `csrc/mainloop_bwd_sm90_fp8r.hpp`,
`csrc/fp11_delta_v6.h`); `DM_XDEFS` (extra -D, stamped into the build info) in `delta_matching/_buildcfg.py` builds a
variant. With no switch set the sources compile the accepted code.

Tools (outside the repo, `kda-work/p1packed-aux/`): `jobs/run.sh` (1-GPU sbatch wrapper), `jobs/build.sh TAG XDEFS [DBG]`
(build into `jit/TAG`, freeze a root in `roots/TAG`), `jobs/screen.sh TAG` (`ab.py --grid one --smoke`, 2 alternating reps,
B2 N8192 causal matched at 16/4 and 32/4, A = frozen `r17_p1`), `jobs/ngate.sh` (`gate.py --mode numerics`, all four
routes, residual-v4 with the LSE +-1 ulp envelope roots and defect repair, as kda-sm90 runs it), `rowdiag.py` /
`margins.py` (per-row residual diagnostics from the gate's row sidecars), `micro/` (SASS / throughput microbenchmarks).

## Where the per-element instructions are (accepted r17_p1, causal pass-1 symbol, unmasked step loop)

Static loop 1032 SASS instructions per warp per 64 x 64 warpgroup step (32 S / dP elements per thread):

| per element (x32 per step) | instructions | notes |
|---|---|---|
| exp line | FFMA, FADD (+8), FMNMX (F1 clamp), MUFU.EX2 = 4 | + 0.5 per row-element share of the per-row FMUL lse*log2e |
| dS line | FFMA (dP descale - delta), FMUL (x P) = 2 | must stay fp32: it is the zero-sum line |
| psi amax | FMNMX = 1 | |
| psi scale + e4m3 | FMUL + 0.5 F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C = 1.5 | |
| P~ e4m3 | 0.5 F2FP | |
| dK fold | 2 FFMA (64 per step) | |
| dV window rescale | 1 FMUL average (64 every 2nd step) | already in the S-retire bubble (S17 RESCALE-EARLY) |
| dV promote | ~1 (LDS + FFMA + STS + re-zero, every 8th step) | |

The bitwise per-element arithmetic is at its floor: every op above is one SASS instruction with no redundant moves,
the e4m3 conversions already convert two fp32 per instruction, and fp32 add / max of a constant cannot be merged
bitwise (fminf(fl(x + 8), 12) == fl(fminf(x, 4) + 8) holds but is still two ops).

## Measurements that bound the levers

- sm_90a codegen (`micro/t.cu`, cuobjdump): `ex2.approx.f16x2` -> 2 x `MUFU.EX2.F16` + PRMT; `ex2.approx.ftz.bf16x2` ->
  2 x `MUFU.EX2.BF16` + PRMT; `cvt.rn.satfinite.e4m3x2.f16x2` -> one `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C` (the same
  count as from an fp32 pair); `fma.rn.f32.f16` (mixed precision) is rejected by ptxas ("requires sm_100").
  There is no packed fp32 ALU on sm_90 (FFMA2 / FMUL2 are sm_100).
- MUFU throughput (`micro/mufu.cu`, job 21178280, 8 independent chains per thread, 132 x 8 x 256 threads): ex2 f32
  15.9, f16 16.0, f16x2 16.0, bf16x2 16.0 ELEMENTS / clk / SM: the packed forms halve the instruction count but not the
  MUFU time; MUFU is 16 elements / clk / SM in every precision.
- Timing probes (job 21178461, wrong outputs, `KDA_PK_PROBE`, screen vs r17_p1, pass 1 H16 / H32):
  - no dV window rescale (-64 FMUL every 2nd step): **+2.48% / +2.54% (slower)**: the rescale fills the S-retire bubble;
  - no psi multiply before the dS' conversion (-32 FMUL per step, 1 instruction per element in the conversion phase):
    +0.50% / -0.50% (noise).
  So outside the exp phase, removing per-element instructions buys nothing measurable; the exp phase is where an
  instruction per element is worth ~1.3-1.5% (EXPB below).
- Numerics structure (pass 0, `fp11_delta_v6.h`): delta = sum_j P^_j dP_j 2^-8 (not normalized by sum_j P^_j), so
  rowsum(dS) = 0 holds only while sum_j P^_j = 2^8 to fp32 round-off, i.e. while P^ is fp32-accurate and consistent
  with the forward LSE. Any f16 / bf16 P^ (rel. error ~2^-11 / 2^-8) breaks the identity by orders of magnitude more
  than k_r 2^-23; the dS line (dP descale - delta) is a cancellation and must stay fp32 for the same reason.

## Candidates

### EXPB (KDA_PK_EXPB=1): the 2^8 offset folded into the per-row bias, pass 0 + pass 1 identically: REJECTED (2 rows)

- Change: b = fma(lse, log2e, -8) per row; P^ = exp2(min(fma(s, dsc, -b), 12)) (was exp2(min(fl(fl(s dsc - fl(lse
  log2e)) + 8), 12))): -1 FADD per element in both passes. Output-changing (P^ rounding, hence delta and everything
  downstream). Commit 20548fc (switch).
- Resources: REG 168 / STACK 0 / 8 (pass 1), DeltaV6 REG 168 / 96, STACK 0, LOCAL 0, no diagnostics; DeltaV6 code
  28416 -> 26368 B.
- Screen (job 21178102, vs r17_p1): pass 0 **-3.96% / -3.91%**, pass 1 **-1.51% / -1.24%** (H16 / H32), others within
  noise. Variant EXPB=2 (b = fl(fl(lse log2e) - 8), FMUL + FADD per row): pass 0 -3.79 / -3.43%, pass 1 -0.80 / -0.14%.
- Numerics (job 21178280, `runs/ngate_expb1_1`, default route): pytest passes; deterministic; per-output mean errors
  all DOWN (dQ 0.2553092 -> 0.2553092 (-4.6e-8), dK -8.0e-8, dV -2.2e-8, O unchanged); per-case changes within 1e-5;
  exemption sets unchanged; **2 well-conditioned rows regress**: d128_gqa4 bh 3 row 1 (2 visible keys) residual
  4.2e-8 -> 4.1e-6 (tolerance 2.4e-7 + LSE envelope 0) and d128_bhnd bh 5 row 1 (2 keys) 2.0e-7 -> 1.35e-6. Rows
  better / worse: gqa4 17878 / 14725, bhnd 4486 / 3638. Margin by visible keys over all D128 matched cases (max of
  candidate - accepted - k 2^-23 - envelope): 2 keys +3.8e-6 (2 regressions), 3-8 keys -4.3e-8, 9-64 -1.0e-6, 65-128
  -7.7e-6, 129-512 -1.5e-5, > 512 -6.1e-5.
- Verdict: REJECTED under residual-v4 as written (strict per-row; the LSE envelope is exactly 0 on those 2-key rows,
  where fl(lse log2e) absorbs the +-1 ulp LSE move). (kda-pass1 rejected the same idea before the envelope existed,
  with 22 rows; the envelope removes all but these two.)

### EXPBX family: EXPB off the diagonal tiles (KDA_PK_EXPB = 3, 4, 5, 6)

Design: keep the certified exp line on the diagonal tiles, where the rows with the fewest visible keys (the smallest
per-row tolerance k_r 2^-23) live, and use EXPB everywhere else; pass 0 and pass 1 apply the same geometric predicate,
so both form the same P^ and delta stays matched. The rationale is the geometry of the tolerance (EXPB's margin grows
with k: see the EXPB table), not the identity of the two failing rows.

| variant | certified-line region | pass 0 implementation | pass 0 W4 loop (static SASS) |
|---|---|---|---|
| 3 | pass 1's masked range (diagonal 128 x 128 + tail) | per-n-block runtime branch (two exp copies) | 928 |
| 4 | same as 3 | stage loop split at a closed-form stage index (verified by brute force vs pass 1's bound) | 699 + 697 (two loops) |
| 5 | same as 3 | one loop; per-stage bias select + warp-uniform +8 branch | 777 |
| 6 (EXPBX64) | pass 0's own masked tiles (64 x 64 diagonal + tail); pass 1 per warpgroup in its masked loop | one loop; need_mask (already computed) selects the bias and the +8 branch | 726 (64 of them the branched FADDs) |

(accepted r17_p1: 715; EXPB=1: 651.) In 4 and 5 the extra live stage index costs ptxas the 8 hoisted wgmma descriptor
pairs: it reuses one uniform register pair with 2 UMOVs before every QGMMA (+23 UMOV, +14 UIADD3, ... per stage). 6 adds
no loop state and keeps the descriptors hoisted (UMOV / UIADD3 counts equal to EXPB=1).

Screens (pass 0 / pass 1, H16 / H32, vs r17_p1): 3: +1.95 / +1.36%, -1.71 / -2.29%; 4: -2.95 / -3.19%, -1.62 / -1.66%
(rerun -3.24 / -3.82%, -1.77 / -2.39%); 5: -2.34 / -3.68%, -1.53 / -2.66%; 6: -3.52 / -3.64%, -1.37 / -1.70% (EXPB=1 in
the same job: -4.19 / -4.46%, -1.65 / -1.80%).

- EXPB=3, numerics gate (job 21178591, `runs/ngate_expb3_1`): **PASS on all four routes** (ac23_eligible, residual-v4
  with envelope and defect repair); mean errors dQ -4.35e-8, dK -1.22e-7, dV -1.93e-8, O 0; worst per-case rise 4.7e-6
  of the error; rows with <= 128 visible keys unchanged (moved 0), rows 129-512 keys max excess -1.5e-5, > 512 -6.1e-5;
  0 regressions. Paper A/B (same job): pass 1 -2.34%, pass 0 +2.08% (the per-n-block branch), e2e fwd+bwd -0.76% ->
  target pass 0 fails.
- EXPB=4, numerics gate (job 21178963, `runs/ngate_expb4_1`): PASS on all four routes (same P^ as 3); paper A/B pass 0
  **-3.58%** geomean (target met), pass 1 -1.77%, e2e fwd+bwd -1.52%, no shape slower, bench D128 matched -0.34%.
  Confirmation (job 21179399, `runs/abpaper_expb4_2`): pass 0 **-2.98%** geomean (-3.34% by sum), pass 1 -1.67%,
  e2e fwd+bwd -1.17%, no shape slower -> misses 3% by 0.02 in the second job. Candidate branch `kda-sm90-p1packed-c1`
  (af57474, default 4) was pushed before the confirmation; superseded by 6 if 6 passes (below).
- EXPB=6: numerics gate + paper A/B in flight.
