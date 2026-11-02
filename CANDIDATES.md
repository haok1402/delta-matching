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

### EXPBX (KDA_PK_EXPB=3): EXPB off the diagonal 128 x 128 block

- Change: the EXPB line on every (64-row m-block, 128-key block) tile outside pass 1's masked range
  (m_block >= ((n128 + 1) 128 - (seqlen_k - seqlen_q) + 63) / 64, INT_MAX for the seqlen_k tail block), the certified
  line inside it; pass 1 selects by loop (masked loop = certified, unmasked loop = EXPB, compile-time), pass 0 by the
  same predicate per 64-key n-block (`kda_expbx_mask_end`). Every row with <= 128 visible keys has all its keys in its
  diagonal block, so its P^, delta and residual are bitwise the accepted ones; rows with >= 129 keys have a per-row
  tolerance >= 1.5e-5, which EXPB's movement never came near (margin <= -1.5e-5 above). Rationale is the geometry of
  the per-row tolerance, not the identity of the failing rows. Commit (see git log, "EXPBX").
- Status: screen + numerics gate in flight.
