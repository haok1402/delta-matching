# kda-p1packed candidates (pass 1 at head dim 128: per-element arithmetic)

Helper kda-p1packed. Branch `kda-sm90-p1packed`, rebased onto `origin/kda-sm90` 0379369 (r17_p0s accepted: forward C3g
plus the pass-0 stack P0NML / 104-24 / P0ALL). The first campaign ran on 9be3622 (r17_p1) and is archived on
`kda-sm90-p1packed-r1`: EXPB variants 1-6 and probes 1-2. Its numbers are against frozen r17_p1. The numbers for the
rebased C3 are against r17_fwd (numerics) and r17_p0s (timing).

The lever is the per-element arithmetic of the pass-1 step:
- packed two-wide math;
- ex2 in f16x2;
- folding the scaling into fewer FFMA;
- e4m3x2 conversions from packed registers;
- per-element conversions between the S / P / dP / dS stages;
- hand-written PTX.

Every lever is a compile-time switch, default OFF: `KDA_PK_EXPB` and `KDA_PK_PROBE` in
`csrc/mainloop_bwd_sm90_fp8r.hpp` and `csrc/fp11_delta_v6.h`. `DM_XDEFS` (extra -D defines, stamped into the build
info, in `delta_matching/_buildcfg.py`) builds a variant. With no switch set, the sources compile the accepted code.

Tools live outside the repo, in `kda-work/p1packed-aux/`:
- `jobs/run.sh`: 1-GPU sbatch wrapper.
- `jobs/build.sh TAG XDEFS [DBG]`: builds into `jit/TAG` and freezes a root in `roots/TAG`.
- `jobs/screen.sh TAG`: `ab.py --grid one --smoke`, 2 alternating reps, B2 N8192 causal matched at 16/4 and 32/4.
- `jobs/abpaper.sh TAG TARGET RUN`: `ab.py --grid paper`, 3 reps, with acceptance predicates.
- `jobs/ngate.sh`: `gate.py --mode numerics` with the four routes, residual-v4, the LSE ±1 ulp envelope and defect
  repair (as kda-sm90 runs it).
- `rowdiag.py` / `margins.py`: per-row residual diagnostics from the gate's row sidecars.
- `micro/`: SASS and MUFU throughput microbenchmarks.
- `sass/`: SASS dumps, loop extraction and opcode histograms.
- Every run directory is under `runs/`.

## ACCEPTED

### C3: EXPBX64 -- ACCEPTED by kda-sm90 as 7a06674 (gate 21181579: residual-v4 PASS on 4 routes vs r17_p0s, pass 0 -4.50%, bench -1.4%; confirmation 21181901: -4.04%; new accepted root r17_pk). Branch rebased on 7a06674; further A/Bs go against r17_pk.

### (original hand-off record) C3: EXPBX64, the 2^8 offset folded into the exp bias off the masked tiles (pass 0 + pass 1; named numerics change)

- **Branch:** `kda-sm90-p1packed-c3` (c0e9f43) is `kda-sm90-p1packed` with `KDA_PK_EXPB` defaulting to 6, on
  `origin/kda-sm90` 0379369.
- **What it changes:**
  - On every (64-row, 64-key) tile that pass 0 does not mask: P^ = exp2(min(fma(s, dsc, -b), 12)), with the per-row
    bias b = fma(lse, log2e, -8). This is one FADD per element less in both passes.
  - On the masked tiles (the 64 x 64 diagonal tile and the seqlen_k tail) the certified line stays:
    exp2(min(fl(fl(s dsc - fl(lse log2e)) + 8), 12)).
  - Pass 0 selects the line with its existing `need_mask`; the P0NML mask-free loop takes the folded line at compile
    time.
  - Pass 1 uses the folded line in its unmasked loop. In its masked loop it applies pass 0's predicate per warpgroup
    (each warpgroup owns one 64-key block).
  - Both passes therefore form the same P^, and the matched delta still comes from the kernel's own P / dP pair.
- **Why the diagonal tiles keep the certified line:** EXPB everywhere (variant 1 below) fails only on two rows with
  2 visible keys. Its per-row margin grows with the key count k, because the tolerance is k 2^-23. Rows with ≤ 64 keys
  have all their keys in the diagonal tile, so they stay bitwise the accepted ones. The choice follows the geometry of
  the tolerance, not the identity of the failing rows.
- **Not bitwise.** Numerics (job 21180811, `runs/ngate_x64_2`):
  - Setup: `gate.py --mode numerics` against frozen r17_fwd, with its DBG build and LSE controls r17_fwd_dbg_lsep / lsem.
    r17_p0s reproduces r17_fwd's outputs bitwise. Against r17_p0s itself the identity check refuses r17_fwd_dbg (code
    sha), and no r17_p0s DBG build is frozen.
  - **PASS / ac23_eligible on all four routes.** pytest passes; outputs are deterministic; the exemption sets are
    unchanged.
  - 0 well-conditioned row regressions. Maximum excess over the tolerance by visible keys:
    - ≤ 64 keys: rows unchanged (moved 0);
    - 65-128 keys: -7.7e-6;
    - 129-512 keys: -1.5e-5;
    - > 512 keys: -6.1e-5.
  - Mean error change per output: dQ +3.7e-7, dK +3.75e-7, dV -5.5e-8, O 0. This is inside the 1e-6 mean-noise
    allowance; on the r17_p1 base the same change lowered all three means.
  - Worst per-case error ratio is 1 + 1.1e-4, an absolute rise below 1e-5.
- **Timing:** `ab.py --grid paper`, 3 reps against frozen r17_p0s, two fresh jobs (`runs/abpaper_x64_p0s_{1,2}`):

| | job 21180824 | job 21180825 |
|---|---:|---:|
| pass 0 geomean (sum) | **-4.53%** (-4.86%) | **-4.69%** (-4.83%) |
| pass 1 (sum) | -1.32% | -1.42% |
| pass 2 / fwd | +0.04% / +0.58% | +0.07% / +0.20% |
| e2e fwd+bwd | -1.28% | -1.43% |
| per shape fwd+bwd (16/4 64K, 32/4 64K, 16/4 8K, 32/4 8K) | -1.54 / -1.19 / -1.19 / +0.04% | -1.51 / -1.40 / -0.96 / -1.05% |
| shapes slower beyond spread | none | none |

  The numerics job (vs r17_fwd) also gave bench D128 matched -1.0%. The fwd entries are an unchanged binary.
- **Resources:** REG 168 (pass 1; DeltaV6 2-WG) / 96 (DeltaV6 W4), STACK 0, LOCAL 0, no ptxas spill / C7510 / C7511 /
  serialization diagnostics. The pass-0 W4 loop keeps its 8 hoisted wgmma descriptor pairs (see variants 4 / 5 below).
- **Status: READY at hand-off; ACCEPTED (7a06674).** The target kernel (pass 0) is ≥ 3% faster on the four shapes in two jobs, and residual-v4 passes
  on all four routes. Pass 1 gets a further 1.3-1.4%.
- **Superseded:** branches `kda-sm90-p1packed-c1` (EXPB=4) and `kda-sm90-p1packed-c2` (EXPB=6), both on the old r17_p1
  base, are replaced by C3.

## Where the per-element instructions are (accepted r17_p1, causal pass-1 symbol, unmasked step loop)

The static loop has 1032 SASS instructions per warp per 64 x 64 warpgroup step, with 32 S / dP elements per thread:

| per element (x32 per step) | instructions | notes |
|---|---|---|
| exp line | FFMA, FADD (+8), FMNMX (F1 clamp), MUFU.EX2 = 4 | plus the per-row FMUL lse*log2e, 0.5 per element |
| dS line | FFMA (dP descale - delta), FMUL (x P) = 2 | |
| psi amax | FMNMX = 1 | |
| psi scale + e4m3 | FMUL + 0.5 F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C = 1.5 | |
| P~ e4m3 | 0.5 F2FP | |
| dK fold | 2 FFMA (64 per step) | |
| dV window rescale | 1 FMUL average (64 every 2nd step) | already in the S-retire bubble |
| dV promote | ~1 (LDS + FFMA + STS + re-zero, every 8th step) | |

The bitwise per-element arithmetic is at its floor:
- Each op above is one SASS instruction, with no redundant moves around it.
- The e4m3 conversions already convert two fp32 values per instruction.
- An fp32 add of a constant and a max cannot be merged bitwise. The identity fminf(fl(x + 8), 12) ==
  fl(fminf(x, 4) + 8) holds, but it is still two ops.

## Measurements that bound the levers

- **sm_90a codegen** (`micro/t.cu`, cuobjdump):
  - `ex2.approx.f16x2` compiles to 2 x `MUFU.EX2.F16` + PRMT.
  - `ex2.approx.ftz.bf16x2` compiles to 2 x `MUFU.EX2.BF16` + PRMT.
  - `cvt.rn.satfinite.e4m3x2.f16x2` is one `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C`, the same count as from an fp32
    pair.
  - Mixed precision `fma.rn.f32.f16` is rejected by ptxas: "requires sm_100".
  - There is no packed fp32 ALU on sm_90; FFMA2 / FMUL2 are sm_100.
- **MUFU throughput** (`micro/mufu.cu`, job 21178280):

  | form | elements / clk / SM |
  |---|---:|
  | ex2 f32 | 15.9 |
  | ex2 f16 | 16.0 |
  | ex2 f16x2 | 16.0 |
  | ex2 bf16x2 | 16.0 |

  The packed forms halve the instruction count but not the MUFU time.
- **Timing probes** (wrong outputs; `KDA_PK_PROBE`; pass 1 H16 / H32):

  | probe | what it removes or changes | base, job | pass 1 |
  |---|---|---|---|
  | 1 | the dV window rescale | r17_p1, job 21178461 | **+2.48% / +2.54%, slower** (the rescale fills the S-retire bubble) |
  | 2 | the psi multiply before the dS' conversion (1 instruction per element) | r17_p1, job 21178461 | +0.50% / -0.50%, noise |
  | 4 | runs the exp line in f16x2: per row, pack the two keys' fp32 arguments, then HADD2 +8, HMNMX2 clamp, ex2.approx.f16x2, unpack | r17_p0s, job 21180811 | **+1.62% / +2.69%, slower**; unmasked loop 1032 -> 1048 static instructions |

  Outside the exp phase, removing per-element instructions buys nothing measurable. In the exp phase, one instruction
  per element is worth about 1.3-1.5% of pass 1 (EXPB).
- **Numerics structure** (pass 0, `fp11_delta_v6.h`):
  - delta = sum_j P^_j dP_j 2^-8. It is not normalized by sum_j P^_j.
  - So rowsum(dS) = 0 holds only while sum_j P^_j = 2^8 to fp32 round-off, i.e. while P^ is fp32-accurate and
    consistent with the forward LSE.
  - Any f16 / bf16 P^ (relative error ~2^-11 / 2^-8) breaks the identity by orders of magnitude more than k_r 2^-23.
  - The dS line (dP descale - delta) is a cancellation and must stay fp32 for the same reason.
  - The residual the gate checks is computed from P^, dP and delta (the DBG rows). The production dS' line and the dV /
    dK accumulation affect only the fp32 errors.

## Variants, in order (first campaign on r17_p1, `kda-sm90-p1packed-r1`)

| variant | change | pass 0 / pass 1 screen (H16, H32) | numerics | verdict |
|---|---|---|---|---|
| EXPB=1 | b = fma(lse, log2e, -8), EXPB on every tile | -3.96 / -3.91%, -1.51 / -1.24% (job 21178102) | job 21178280: 2 regressions, both row 1 (2 keys, LSE envelope 0): gqa4 4.2e-8 -> 4.1e-6, bhnd 2.0e-7 -> 1.35e-6; all mean errors down; rows better / worse gqa4 17878 / 14725 | REJECTED (strict per-row) |
| EXPB=2 | b = fl(fl(lse log2e) - 8) (host-precomputable form) | -3.79 / -3.43%, -0.80 / -0.14% | not run | weaker than 1 |
| EXPB=3 | certified line on pass 1's masked range (128 x 128 diagonal + tail); pass 0 per-n-block runtime branch | +1.95 / +1.36%, -1.71 / -2.29% | job 21178591: PASS on 4 routes; rows ≤ 128 keys unchanged | REJECTED on timing (paper: pass 0 +2.08%, pass 1 -2.34%) |
| EXPB=4 | as 3; pass 0 stage loop split at a closed-form index (checked by brute force against pass 1's bound) | -2.95 / -3.19% (rerun -3.24 / -3.82%), -1.62 / -1.66% | job 21178963: PASS on 4 routes | paper pass 0 -3.58% then **-2.98%** (confirmation 21179399): missed 3% in the second job |
| EXPB=5 | as 3; one loop with a per-stage bias select and a warp-uniform +8 branch | -2.34 / -3.68%, -1.53 / -2.66% | (same P^ as 3) | no better |
| EXPB=6 (EXPBX64) | certified line exactly on pass 0's masked 64 x 64 tiles; pass 1 per warpgroup | -3.52 / -3.64%, -1.37 / -1.70% | job 21179856: PASS on 4 routes; rows ≤ 64 keys unchanged; 65-128 keys -7.7e-6 | on r17_p1: pass 0 -3.82% / -3.66% (jobs 21179856 / 21180390); rebased as C3 |

Hand-PTX finding (variants 4 and 5):
- With an extra live stage index in the pass-0 W4 loop, ptxas stops hoisting the 8 wgmma descriptor pairs.
- Instead it reuses one uniform register pair, with 2 UMOVs before every QGMMA: +23 UMOV, +14 UIADD3, +8 ULOP3 and
  +8 ULEA / USHF per stage. The loop has 699 static instructions vs 651 for EXPB=1.
- Variant 6 adds no loop state (it reuses `need_mask`) and keeps the descriptors hoisted.
- After rebasing, the P0NML loop split lets the mask-free loop take the folded line at compile time. Pass 0 then
  screens at -4.43 / -4.60% (job 21180541).

## Lever coverage (each lever: ≥ 2 designed variants, measured)

| lever | variants | outcome |
|---|---|---|
| packed two-wide math (HFMA2 / HMUL2 / HMNMX2) | (a) exp line in f16x2 (probe 4); (b) dS' line packed after the fp32 FFMA (pack P^ and D, HMUL2, HMNMX2 amax, HMUL2 psi, F2FP from f16x2: -0.5 instructions per element at best) | (a) +1.6 / +2.7% slower and breaks the identity; (b) bounded by probe 2: removing a full instruction per element in that phase is noise, and f16 range / rounding would cost dK / dQ error |
| ex2.approx.f16x2 / bf16x2 | f16x2, bf16x2 (microbenchmark + SASS); f16x2 in kernel (probe 4) | MUFU is 16 elements / clk / SM in every precision; the packed forms are 2 MUFU + PRMT; slower in kernel |
| fold the scaling into fewer FFMA | EXPB 1, 2; EXPBX 3, 4, 5, 6 | **C3 (EXPBX64) READY**: pass 0 -4.5 / -4.7%, pass 1 -1.3 / -1.4% |
| cvt e4m3x2 from packed registers | P~ from f16x2 (in the probe-4 design), dS' from f16x2 (PKDS design) | the same instruction count as from fp32 pairs; packing costs what it saves; no gain to take |
| per-element conversions between stages | per-row lse*log2e folded into the bias FFMA (part of EXPB); the dV window rescale removed (probe 1) | the first is inside C3; the second is slower (+2.5%) |
| hand PTX | the exp clamp / offset algebra (no count change, proved); the explicit one-line exp with a uniform +8 branch and the descriptor-hoisting fix (variants 5 / 6) | the second is C3's pass-0 form |

## Final summary (helper kda-p1packed)

- **Deliverable:** C3 (EXPBX64), READY.
  - Pass 0 -4.53% / -4.69% geomean, pass 1 -1.32% / -1.42%, e2e fwd+bwd -1.28% / -1.43%, in two fresh jobs against
    r17_p0s.
  - Residual-v4 PASS on all four routes (job 21180811).
  - It is a named numerics change; its commit carries the evidence.
- **Why packed half-precision math has no room in pass 1 on sm_90.** The measurements above show it from three sides:
  - The MUFU unit is 16 elements / clk / SM in every precision.
  - sm_90 has no packed fp32 ALU, and every f16x2 / bf16x2 path pays a pack per pair that cancels its saving.
  - The zero-sum line (exp, dS, delta) needs fp32 P^ because delta is not normalized by sum P^.
- **The pass-1 step is not issue-bound per element.** Removing a full instruction per element outside the exp phase
  is noise (probe 2), and removing the rescale FMULs is slower (probe 1). Only the exp phase responds to instruction
  cuts: about 1.4% per instruction per element.
- **The bitwise per-element arithmetic is at its floor** (table above).
- **What is left for pass 1** lies outside per-element arithmetic, as kda-pass1 and kda-p1redesign found:
  - synchronization (psi rendezvous, dS' store);
  - the register budget that blocks overlap (dK 64 + dV 64 + dK temp 64 + S / dP 64 of 240);
  - structural changes to the dataflow.
- **A follow-up others could use.** The geometric restriction used here generalizes: keep the certified rounding on
  the diagonal tiles, where per-row tolerances are smallest. This is the same principle as the forward's accepted
  C3g, which applies FWDDFOLD on unmasked steps only. It could rescue other output-changing levers that fail
  residual-v4 only on few-key rows, e.g. pass-0 LAZY / DEFER, if their failing rows have ≤ 64-128 keys. I did not
  measure this.
