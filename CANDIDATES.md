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

## The ledger: an FA3-fp8 skeleton plus each delta-matching addition (switch `KDA_SK`, TIMING ONLY)

The skeleton is our kernel with every delta-matching addition taken out:
- no psi: no amax / shuffles / slots / CTA barrier, dS' = e4m3(dS);
- dK accumulated in place: no temp, no fold;
- no dV window: no rescale, no promote;
- no dS' transpose / stmatrix / gmem store.

What remains is FA3's BF16 backward with fp8 operands, block scales and the matched exp / dS line.

All probes were checked in SASS to keep the full 8 + 16 QGMMA set. The first round of SK4 / SK15 / SK16 lost the dV
gemm: with the promote gone, ptxas dummied the dead dV wgmmas as `HGMMA.64x8x16 RZ`. Those probes now finish with one
final promote, and the numbers below are the corrected ones (job 21203349 and later). Columns are pass 1 vs r19_fwd,
B2 N8192 16/4 / 32/4, 2 alternating reps (`ab.py --smoke`).

| build | what | pass 1 |
|---|---|---:|
| sk15 | skeleton (all four additions removed) | **-29.5% / -30.3%** |
| sk14 | skeleton + psi | -21.9% / -22.4% |
| sk13 | skeleton + dK temp / fold | -26.8% / -27.8% |
| sk11 | skeleton + dV window | -22.9% / -23.4% |
| sk7 | skeleton + dS' transpose / store | -21.6% / -22.5% |
| sk1 | full minus psi | -10.8% / -10.6% |
| sk8 | full minus dS' transpose / store | -8.1% / -8.3% |
| sk4 | full minus dV window | -8.7% / -7.8% |
| sk16 | full minus dV promote only | -7.5% / -6.0% |
| sk32 | full minus dV rescale only | +0.4% / +0.0% |
| sk2 | full minus dK fold (in-place dK) | -4.2% / -4.5% |

**Reading.**
- The FA3-fp8 skeleton is the ceiling of any FA3 port: 30% faster than r19_fwd.
- The cost above it splits into four parts:
  - psi, about 11%;
  - the dS' hand-off to pass 2, about 8%;
  - the dV window, about 8%, almost all of it the promote;
  - the dK fold, about 4%.
- psi, the fold and the dS' codes are the matched method's numerics contract. Their cheaper forms were measured by
  earlier helpers:
  - per-WG psi: pass 2 +7..9%;
  - RIP / DKACC: slower, or fail residual-v4;
  - DSNAT: 0%.
- The one part that showed headroom is the dV promote. Its two variants:

### REJECTED: PE, promote deferred into the next step (bitwise)

- Switch `KDA_PE=1` (default 0).
- What it changes: a window promote that is not the tile's last runs right after the next step's S / dP issue, in the
  S-retire bubble where RESCALE-EARLY paid, instead of at the end of its own step where nothing is in flight.
- Bitwise: dQ / dK / dV are bitwise on the 4 dump shapes (job 21203555).
- Resources: REG 168, STACK 0.
- Timing: pass 1 +0.51% / -0.65%. Neutral.
- Reason: the promote's cost is not exposed latency. It is throughput: 64 KB of LDS + 64 KB of STS per CTA per promote
  (both WGs at once), against the wgmma smem operand reads.

### REJECTED: PT32 / PT16, dV promote window 8 -> 32 / 16 m-tiles (named numerics change, dV only)

- Switch `KDA_PT=32`.
- What it changes: dV promotes into the fp32 smem overlay every 32 m-tiles (2048 rows) instead of 8. The rescale chain
  inside a window stays clamped exactly as before. dQ, dK, every dS row residual, psi and the dS' codes are bitwise
  unchanged, so residual-v5's row check is unaffected. Only the dV error vs fp32 can move.
- Screen, pass 1 (job 21203949): 8K 16/4 -3.80%, 8K 32/4 -4.63%, 64K 16/4 -3.24%.
- PT16: -2.35% / -0.49% / -2.35%.
- dV rel-L2 error vs fp32 on the 4 dump shapes:

  | build | case 1 | case 2 | case 3 | case 4 | verdict |
  |---|---|---|---|---|---|
  | PT32 | 4.8325e-2 -> 4.8290e-2 | 4.9126e-2 -> 4.9123e-2 | 1.3031e-1 -> 1.3029e-1 | unchanged | lower or equal |
  | PT16 | 4.8325e-2 -> 4.8338e-2 | | | | +1.3e-5 on case 1, over the 1e-5 per-case tolerance: rejected |

- Resources: REG 168, STACK 0, no spills.
- Status: **REJECTED.** This is the same lever as kda-pass1's PW16 / PW32 (origin/kda-sm90-pass1-v2), which already
  failed the per-case dV tolerance on the 72-case default route: d128_bhnd dV +1.67e-5, gqa3_promote dV +3.3e-5. PT32 is
  the same code change, so there is no new reason to expect a different verdict. PT16 already fails it on my own dump
  shapes. The paper-grid A/B pair (jobs 21203959 / 21203960) was cancelled once this was found.
- Even if the numerics passed, PT32 gives -3..-5% on pass 1, under the 10% bar for this brief.

## Final summary (kda-bwdport)

**No READY candidate.** No FA3-derived design comes within reach of a 10% gain on pass 1 or on pass 1 + pass 2.

**What FA3's backward offers here.** The accepted pass 1 already has FA3's hdim-128 backward geometry and step order:
- 64 x 128 tiles, 2 MMA warpgroups, 24 / 240 registers, a TMA producer warp;
- SdP_swapAB and RS dV / dK.

What it lacks:
- **FA3's in-kernel dQ** is a measured loss in every form tried:
  - The 5th gemm alone costs +8..+12% of pass 1.
  - The hand-off costs ~+9% more.
  - FA3's deterministic ordering costs +34% more.
  - The best non-deterministic form is +3% vs pass 1 + pass 2, and the deterministic form is +34%.
  - The reason: FA3 BF16 is tensor-bound, so its 5th gemm hides under the gemms. Our fp8 step is bound by the vector
    stream, and pass 2, the work being removed, is a cheap bytes-bound kernel at 94% of its floor.
- **FA3's simpler numerics** (no psi, no dK fold, no dV window, no dS' hand-off) are worth 30% of pass 1 together. That
  is the skeleton bound, and the ceiling of any port. Each piece is part of the matched method's numerics contract:
  - psi, about 11%;
  - the dS' hand-off, about 8%;
  - the dV window, about 8%;
  - the fold, about 4%.

**Levers and variants tried here:**
- **dQ fusion**: dQ issued at the start or end of the step; smem + bulk-reduce writer warp vs register atomics;
  non-deterministic vs ordered-semaphore vs lagged-release determinism; SPT dispatch. Analytic: a dQ warpgroup, int64
  exact accumulation. **REJECTED.**
- **Warp specialization**: a dQ writer warp, measured inside FQ. The 4th-warpgroup forms are register-infeasible and
  were measured slower by p1fresh (FR66 / FR67).
- **Tile shapes**: FA3 uses 64 x 128 for causal hdim 128, which is ours. Its non-causal 80-row tile breaks the 64-row
  psi / 128-row q8s block contract. 128-row and 64-key variants were priced by p1redesign / p1fresh and fail.
- **dV window promote**: PE (bitwise, neutral); PT16 / PT32 (dV numerics, a repeat of PW; rejected).
- **psi / dS' / fold**: priced by the skeleton ledger. Their cheaper forms are covered by the earlier helpers'
  measurements (WGPSI, DSNAT, RIP, DKACC), all rejected.

**For whoever continues.**
- The 30% between r19_fwd's pass 1 and the FA3-fp8 skeleton sits in four pieces of the numerics contract, not in the
  schedule.
- A 10% pass-1 gain needs one of two things:
  - a contract change that removes a piece outright, for example psi per larger tile or no dS' hand-off. Such a change
    would have to pass residual-v5 against the original kernel.
  - a cheaper form of the dV window that keeps the dV error inside 1e-5 per case. The promote's cost is smem
    bandwidth (128 KB per CTA per promote), so the form has to move fewer bytes, not just fewer instructions.
- All switches (`KDA_FQ`, `KDA_FQX`, `KDA_FQPOS`, `KDA_SK`, `KDA_PE`, `KDA_PT`) default to the accepted kernel. FQ, SK
  and PT builds are timing or numerics probes; do not gate them.

## Re-read under docs/kda-sm90/TASK.md (kda-sm90 after e28c03e)

Nothing here changes status under the new rules:
- **Stacking (bitwise or residual-v5 gains >= 1% in two jobs):** this branch has no such piece. PE is the only
  bitwise variant and it is neutral (+0.51% / -0.65%).
- **Summed time of every touched kernel:** FQ was already judged on pass 1 + pass 2 together (+3% non-deterministic,
  +34% deterministic, both slower).
- **Residual-v5:** PT16 / PT32 change dV only, so the dS rows are bitwise. They fail the per-case fp32-error clause:
  PT16 on my dump shapes, PW32 on the 72-case route.

The branch stays DONE with no READY candidate.
