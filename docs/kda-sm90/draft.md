# Draft: evolve the delta_matching sm90 kernels toward speed of light

## Goal

Make the Delta-Matching fp8 attention kernels (forward, matched-delta pass 0, pass 1 dK/dV, pass 2 dQ, the Triton
producers, the twins and the torch glue around them) faster on H100 (sm90) without changing their numerics, one
measured hypothesis per round, on branch `kda-sm90`.

## Where we start (measured, results/baseline.md, commit "Add per-kernel speed-of-light benchmark")

Environment: H100 SXM, driver 535 + cuda-compat 580 (CUDA 13.0.2), torch 2.12.0+cu130, Triton 3.7.0, FA3 built from the
pinned flash-attention commit. Sustained SM clock about 1950 MHz under this load -> dense fp8 peak 2109 TFLOPS; HBM3
3.35 TB/s. Nsight Compute counters are not accessible on this cluster (ERR_NVGPUCTRPERM), so evidence comes from
CUPTI kernel timings (benchmarks/kernels.py), SASS / resource usage (cuobjdump) and controlled A/B builds.

Waste ranking over the main grid (D 128/256, N 2k-32k at 32k tokens, MHA/GQA4, causal matched, causal stale, full stale):

| kernel | measured | SOL | % of SOL | wasted |
|---|---:|---:|---:|---:|
| pass 1 (dK, dV, dS' tiles) | 507.7 ms | 129.9 ms | 26% | 377.9 ms |
| forward | 187.0 ms | 64.7 ms | 35% | 122.3 ms |
| pass 2 (dQ) | 106.1 ms | 62.8 ms | 59% | 43.3 ms |
| torch glue (layout copies, GQA fold) | 36.6 ms | 3.0 ms | 8% | 33.6 ms |
| pass 0 (matched delta) | 36.1 ms | 16.2 ms | 45% | 19.9 ms |
| producers + twins | 27.4 ms | 17.7 ms | | 9.7 ms |

Reference points: FlashAttention-3 fp8 forward is about 25% faster than our forward; FA3's whole bf16 fwd+bwd is faster
than our fp8 fwd+bwd at every shape (for example D128 N16k causal: 12.7 vs 16.3 ms).

## Constraints (the numerics contract)

- README.md and csrc/README.md define the contract. The flag set in delta_matching/_buildcfg.py (enforced by
  csrc/dm_flagset.h) is the kernel's numerical identity. Prefer changes whose outputs are BITWISE identical to the
  baseline library (schedule, layout, launch, addressing changes). If a change alters the certified flag set, the commit
  says so and shows the numerics are unchanged or better.
- Outputs must stay deterministic; the matched-mode identity rowsum(dS) = 0 must hold to fp32 round-off; errors against
  fp32 must not exceed the current values.
- No pushes to main or upstream; the branch kda-sm90 is pushed to origin after every accepted commit.
- All GPU work runs as sbatch jobs (1 GPU, --wait); builds through Slurm.

## Acceptance of a round

1. `python -m pytest tests/` passes on the candidate build.
2. An equivalence check against the baseline library (DM_SO) on a fixed set of shapes (D 128/256, MHA/GQA, causal
   matched/stale, full stale, a held-out shape): O, dQ, dK, dV, LSE bitwise equal, or, if not bitwise, determinism plus
   relative error vs fp32 no worse than baseline and the matched rowsum(dS) residual at fp32 round-off (DM_DBG build).
3. The target kernel is at least 3% faster on the main grid, confirmed by a second fresh job; no regression above 1% on
   the held-out shapes or on benchmarks/bench.py at D 128 and D 256 (end to end).
4. One commit per accepted improvement with the measurement in the message; push to origin kda-sm90.

Stop when two consecutive rounds give no accepted gain, or every kernel is within 10% of its speed of light; then write
results/REPORT.md (per-kernel before/after, % of SOL, what worked, what did not, what remains).

## Hypothesis backlog (ranked by wasted time; one per round, re-ranked after each round)

H1. Pass 1 at hd128 (mainloop_bwd_sm90_fp8r.hpp), schedule overlap inside each tile. Today each warpgroup runs, per
64-row m-block: S and dP wgmma -> exp / dS elementwise -> psi amax -> CTA-wide named barrier (cross-WG psi) -> P and dS'
conversions -> dS' transpose (16 shfl + 16 prmt) -> 2 WG barriers + stmatrix + TMA store -> dV and dK wgmma -> wait ->
dK fold. The wgmma are asynchronous: issue the dV gemm (P^T dO, which does not depend on psi) before the psi barrier,
and run the dS' transpose / stmatrix / TMA store after issuing the dK gemm (pre-packing the transpose words before the
issue, since wgmma A registers must not be read while in flight). Same instructions per element, same order of
arithmetic -> bitwise. Expected: hides the barrier wait and the transpose behind tensor-core time.
H2. Pass 1 at d256 (mainloop_bwd_sm90_fp8v10.hpp): the analogous overlap / barrier reduction, after reading its tile loop.
H3. Forward producers and glue: the Q/K/V quantizers and the k-mean read `x.reshape(BH, N, D).contiguous()` of the
(B, N, H, D) inputs, which is three full transposition copies per call (0.57 ms at 32k tokens, H 16, D 128); read the
strided input directly (as doq_m17 already does for dO). The GQA dK/dV fold `dk.permute(...).reshape(B, Hk, rep, N, D)
.sum(2)` copies before reducing; reduce on the contiguous (B, N, Hk, rep, D) view instead (bitwise: the same fp32 sum of
the same `rep` values per element, verified). Expected about 3-6% of end-to-end fwd+bwd and 15-40% of end-to-end fwd.
H4. Pass 2 (fq8_dq_pass2*.h), 59-76% of the HBM bound at D 128: pipeline depth / occupancy / TMA box shape for the dS'
stream; L2 promotion; schedule only.
H5. Forward (mainloop_fwd_sm90_fp8f.hpp): intra-warpgroup overlap / ping-pong schedule of the per-32-key P basis work,
persistent scheduling details; schedule only.
H6. Pass 0 (fp11_delta_v6.h / fp11_delta_v4.h) analogous schedule work.
H7. Persistent / LPT scheduling and epilogue overlap for pass 1 (non-persistent SingleTileBwdLPTScheduler today).
H8. Launch overhead: twins fused into the producers (DM_FUSED_TWIN path is self-checked), fewer small launches.

## Tooling to add

- benchmarks/equivalence.py: dump O / dQ / dK / dV / LSE for a fixed shape set with a given library (DM_SO) and compare
  two dumps bitwise; also report relative errors vs fp32 and determinism.
- benchmarks/kernels.py and benchmarks/summarize.py (exist): main + held-out grids, before/after comparison.
- results/: each accepted round adds its measured JSON and a summary; results/REPORT.md at the end.
