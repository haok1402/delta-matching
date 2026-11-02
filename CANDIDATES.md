# kda-bwdport candidates (pass 1 / pass 1 + pass 2 at head dim 128, ported from FA3's BF16 backward)

Helper kda-bwdport, branch `kda-sm90-bwdport` (on `origin/kda-sm90` c8c2f48 = r19_fwd accepted). Brief: take FA3's hdim-128
backward mainloop, add only what matched pass 1 needs, price every addition, and keep only designs that can beat the
accepted kernel (frozen `r19_fwd`) by 10% on pass 1, or on pass 1 + pass 2 if dQ is folded in.

## Starting point: our pass 1 already is FA3's hdim-128 backward schedule

FA3's hdim-128 causal backward (`hopper/flash_bwd_launch_template.h`, `run_mha_bwd_hdim128`) uses kBlockM 64, kBlockN 128,
2 MMA warpgroups, SdP_swapAB, Mma_dKV_is_RS, AtomLayoutNdKV 2, a 24 / 240 register split, one TMA producer warp, and the
LPT scheduler. The accepted P1-R collective (`csrc/mainloop_bwd_sm90_fp8r.hpp`, M20) was forked from exactly that
geometry. Its step order is FA3's: S and dP issue, exp, dS, the P codes into the RS dV gemm, the dS codes into the RS dK
gemm. Three things in FA3's backward are not in ours:

1. **dQ in the same kernel.** FA3 adds a 5th gemm per step, dQ = dS K, with dS in smem. The fp32 partial goes through
   smem to a dQ-writer warp, which bulk-reduce-adds it into an fp32 dQaccum in L2. Deterministic mode uses a
   per-(m-block) semaphore in descending n order plus SPT dispatch. A postprocess kernel converts dQaccum to bf16.
2. **dK accumulated in place.** FA3 has no per-tile basis, so it needs no temp and no fold.
3. **No psi tile scale and no dV window.**

(2) and (3) are numerics that the matched method needs. They were priced by earlier helpers: RIP in-place dK, +3.6%;
DKACC, which fails residual-v4; per-WG psi, which costs pass 2 +7..9%. That leaves (1), dQ folded in, as the one
structural FA3 lever never prototyped. p1redesign rejected it on paper, assuming per-tile fp32 partials go to HBM.
FA3 instead reduces them in L2. Below it is built and priced step by step.

Tools live outside the repo, in `kda-work/bwdport-aux/`:
- `jobs/build.sh`: a full JIT build per `name|XDEFS` into `builds/<name>/root`.
- `jobs/tu.sh`: single-TU ptxas.
- `jobs/fqjob.sh`: a dQ numerics dump, then `ab.py --grid one --smoke`, 2 alternating reps, A = frozen r19_fwd.
- `tools/dump.py` / `tools/cmp.py`: grads of `dm_attn_func_bhnd` matched causal on 4 shapes, against an fp32 reference.
- Logs are in `jobs/logs/`; runs are in `runs/j<jobid>/`.

Numbers are the pass-1 (and pass-2) median of B vs A. Rep spreads are <= 1.1%.

## REJECTED: FQ, FA3-style in-kernel dQ (pass 2 folded into pass 1)

Commit cc7924a and later (switches `KDA_FQ`, `KDA_FQX`, `KDA_FQPOS` in `csrc/mainloop_bwd_sm90_fp8r.hpp`, all default 0;
with them off the build is the accepted kernel).

**Design**
- The Kt twin tile (128 d x 128 keys, the tile pass 2 already reads) is loaded once per CTA with K and V.
- dQ(m) = dS'(m) x Kt is an SS wgmma, m64 n64 per warpgroup over K = 128 keys, in two K = 64 calls.
  - It reads the two warpgroups' dS' smem half tiles. These are already parity-double-buffered and already certified
    complete by the psi barrier of step m + 1, so no new barrier is needed.
  - It is issued at the start of step m + 2 (FQPOS 0) or at the end of step m + 1 (FQPOS 1). Those are the only points
    where 32 more accumulator registers fit next to dK / dV / S / dP.
- The partial is scaled by psi(m). k8s is already folded into the codes.
- The partial goes to the fp32 dQaccum [BH, nbm, 2, 4096] in wgmma-fragment order, by one of two hand-offs:
  - a 16-KB smem buffer per warpgroup, then a dQ-writer warp (producer warp 1) doing a bulk reduce-add (FA3's TMA path);
  - float4 atomics straight from registers (FA3's non-TMA path).
- Determinism: a per-(bh, m-block) semaphore with adders in descending n-block order, and the scheduler switched to
  SPT. This is FA3's deterministic causal mode.
- A small postprocess kernel (`fq_dq_post_kernel`) applies sm/256 and writes bf16 dQ. Pass 2 and the dS' gmem store
  are gone.

**Correctness** (fq31, the full deterministic design; also fq31p1 and fq23x129p1; job 21199545)
- dK and dV are bitwise equal to r19_fwd.
- dQ differs by 6-8e-4 relative (bf16 level) because the tile accumulation order changes.
- dQ error vs fp32 is unchanged to 4-5 digits on 4 shapes (e.g. 6.3465e-02 vs 6.3465e-02; scale x3 2.2412e-01 vs
  2.2413e-01).

**Pricing**, B2 N8192 16/4 (and 32/4 where shown). Base pass 1 is 1.66 ms and pass 2 0.43 ms (32/4: 3.33 / 0.83 ms).

| step added | build | pass 1 change | notes |
|---|---|---:|---|
| Kt load + dQ wgmma groups, result unused | fq1 | +3.9% (32/4 +3.6%) | **not the gemm cost**: ptxas replaced the unused QGMMAs by dummy `HGMMA.64x8x16 RZ` (SASS) |
| dQ gemm, result consumed (no STS), start of step | fq3x65 | **+11.6%** (64K: **+8.0%**) | the real cost of the 5th gemm in this step |
| same, end of step | fq3x65p1 | +12.9% (64K +9.3%) | |
| + psi scale + 8 STS.128 of the partial + hand-off barriers + fence | fq3 | +24.3% (32/4 +24.0%) | STACK 8 -> 32 |
| same without barriers / without fence / without STS | fq3x1 / fq3x2 / fq3x4 | +20.6% / +20.8% / +3.9% | the cost is the gemm plus the STS, not the barriers or the fence |
| same, end of step | fq3p1 | +22.3% | |
| + writer bulk reduce-add into dQaccum (L2) | fq7 / fq7p1 | +26.2% / +23.5% (32/4 fq7 +25.2%) | the L2 reduce itself costs ~1% |
| writer releases each WG's buffer separately | fq7x8 | +26.5% | |
| float4 register atomics instead of smem + writer (FA3 non-TMA) | fq23x129 / p1 | **+48.6% / +48.3%** (64K +44%) | |
| + SPT dispatch only (no semaphores) | fq7x32 | +32.5% | SPT tail: +6% |
| + deterministic order (semaphores, descending n) | fq15 | **+61.3%** (32/4 +58.9%) | waiting for the predecessor CTA: +28% more |
| order released one m-block late | fq15x16 / fq15x24 | +60.2% / +61.0% | no help |
| full design, pass 2 replaced by the postprocess | fq31 / fq31p1 | +61.4% / +61.6% | pass 2 0.427 -> 0.113 ms (postprocess); 64K: 12.2 -> 0.47 ms |

**Totals.** Pass 1 + pass 2 (or postprocess), 8K 16/4: base 2.09 ms.

| design | total | change |
|---|---:|---:|
| fq31, deterministic | 2.80 ms | +34% |
| fq7p1 + postprocess, non-deterministic | 2.16 ms | +3% |
| bound: gemm only (fq3x65) + postprocess, free hand-off and free determinism | 1.98 ms | -5.8% |
| same bound at 64K 16/4 | 59.8 + 0.47 vs 67.5 ms | -10.8% |

The unattainable bound averages about -8% over the shapes. Every real form is slower than pass 1 + pass 2.

**Why** (this differs from FA3 BF16, where the 5th gemm pays):
- FA3's BF16 step is tensor-bound: 5 BF16 gemms = 2560 tensor cycles at about 61% utilization. Its vector work hides
  under the gemms.
- Our fp8 step is vector-stream-bound: 4 fp8 gemms = 1024 tensor cycles at about 34%.
- An extra 256-cycle gemm therefore lands on the critical vector stream through the tensor/ALU coupling solcheck
  measured (C_alu + 0.55 C_tensor). It also needs its own hand-off instructions (32 FMUL + 8 STS.128 + waits) in the
  issue-bound MMA warps.
- Pass 2, by contrast, is a bytes-bound kernel already at 94% of its floor. Folding it in trades cheap HBM time for
  expensive MMA-warp time.
- Determinism (FA3's ordered semaphores) adds 34 points of stalls on top.

**Other forms, analytic:**
- **A dedicated dQ warpgroup** (FA3 has none either). It needs a 64-register dQ accumulator plus addressing. The register
  file leaves at most 32 per producer-group thread next to two 240-register MMA warpgroups. p1fresh's FR67 / FR66
  measured fourth-warpgroup gemm roles +8..+22% slower.
- **Exact (order-free) determinism** via int64 fixed-point bulk adds. It needs a per-row grid bound known before pass 1
  (a pass-0 change) and 64 KB of smem per step that does not fit (206 KB used of 227). Even free, it only recovers the
  ordering cost, leaving at best the -6 / -11% bound above.

Status: **REJECTED.** No form reaches -10% on pass 1 + pass 2, and the deterministic form is +34%.
