# Results: kda-sm100 candidates vs FA4 (hd128, GB200) — maintained by fa4-bench

All rows are measured by attnbench.py. The candidate and FA4 are timed in the same process on the same GPU, interleaved
in 6 rounds, with every GPU busy (4 shards). Speedup = FA4_ms / cand_ms, geomean over 12 shapes (H16/16 + H32/8,
N 1k-32k). "e2e" means the time starts from bf16 q/k/v, so the quantization producers are included, as in training.

| date | cand (sha) | pass | causal geomean (min-max) | non-causal geomean (min-max) | FA4 fp8 fwd same run | rel.err O vs fp32 | raw |
|---|---|---|---|---|---|---|---|
| 2026-11-08 17:38 | C1-v1 fwd e2e (811db9e) | fwd | 0.902 (0.624-1.170) | 0.936 (0.705-1.161) | 1.173 / 1.030 | 5.0-5.7e-2 | results/c1v1_fwd.s*.jsonl |
| 2026-11-08 17:53 | B1 fwd+bwd, stale mode, deterministic bwd (kda/cands/b1_cur.py @ wt/kda-sm100 7ed21b0) | fwd_bwd | 0.639 (0.368-1.003) vs FA4; 0.661 vs FA4-det | 0.728 (0.438-1.089); 0.746 vs FA4-det | - | O 5e-2, dQ/dK 6.7e-2, dV 5e-2 | results/b1_stale.s*.jsonl |
| 2026-11-08 18:23 | FULL matched (api.dm_attn_func, pinned snap 6a31419: C1-v2 fwd + P0 kernel + B1 det bwd) | fwd_bwd | **0.729** (0.537-0.891); 0.756 vs FA4-det | 0.771 (0.587-0.945); 0.793 vs FA4-det | - | O 5e-2, dQ/dK 6.5e-2, dV 5e-2 | results/full_6a31419.s*.jsonl |
| 2026-11-08 18:23 | FULL stale (same build) | fwd_bwd | 0.686 (0.385-1.047); 0.711 vs FA4-det | 0.800 (0.462-1.169); 0.823 vs FA4-det | - | same | same |

### Notes on C1-v1 (fwd only, no bwd yet)
- It is faster than FA4 only at long sequences: 1.05-1.17x at 16k-32k, where FA4 hits the power cap and its clock drops.
  It is slower at <= 4k: 0.62-0.72x at 1k. kda's own screening said geomean 0.993 for the kernel alone. Two reasons it
  reads lower here: (1) e2e includes the producer (quant_qkv2), which costs about 0.08-0.1 ms per call (the dominant
  loss at 1k-2k); (2) kda's screening FA4 times at long shapes were 20-29% higher than in this harness
  (e.g. H32/8 N16k non-causal: 6.62 ms in kda/benchmark.csv vs 5.14-5.30 ms here), which inflated its long-shape ratios.
- Error vs the fp32 reference is 5e-2, the same as FA4's own fp8 fwd (5e-2), against 2.3e-3 for bf16. Pass/fail on
  accuracy is the validator's call.
- FA4's fp8 fwd in the same runs: 1.17x causal / 1.03x non-causal. C1-v1 does not beat FA4's own fp8 fwd yet.

### Notes on B1 (stale mode; matched still uses a torch delta, so it was not timed)
- bwd alone: 0.630x causal / 0.745x non-causal vs FA4 (default, non-deterministic bwd). FA4 with deterministic=True
  is 5-6% slower than default on bwd (0.946-0.948x), so the like-for-like ratio (deterministic vs deterministic) is
  0.661x / 0.746x for fwd+bwd. **The headline uses FA4 default (non-deterministic), the stronger baseline**; the
  deterministic comparison is reported next to it.
- Causal fwd+bwd: 1k 0.37-0.43x, 8k 0.70-0.75x, 32k 0.95-1.00x. It reaches parity only at 32k (and 1.09x at 32k
  non-causal, GQA). Short shapes are dominated by fixed costs: host time is about 1.1 ms per fwd+bwd call against
  0.4 ms for FA4 (separate producer launches), plus the producers' GPU time.
- The fwd inside this composite is 0.64x FA4, below C1-v1 e2e (0.90x). b1_cur.py quantizes with quant_rows x3 + kmean
  instead of the fused quant_qkv2.

### Notes on FULL @6a31419 (first complete Delta-Matching sm100 build; matched causal is the headline)
- Matched, causal, fwd+bwd vs FA4: 1k 0.71x / 0.54x (H16/16 / H32/8), 8k 0.74x / 0.74x, 32k 0.86x / 0.89x.
  fwd 0.90x, bwd 0.68x: **the bwd is the gap.** It is 856-1000 TF against FA4's 1250-1380 TF at >= 8k.
- The P0 matched-delta pass costs about 20% of the bwd at long shapes (32k causal bwd: matched 11.0 vs stale 9.0 ms
  for H16/16; 22.2 vs 18.2 for H32/8). At short shapes, stale is SLOWER than matched (1k causal bwd: 1.21 vs 0.76 ms;
  32/8: 2.77 vs 1.89), so the stale delta path has a large fixed cost (probably the unfused torch rowsum(dO*O) plus
  extra producers).
- Stale wins at 32k only: 1.05x causal H32/8, 1.17x non-causal H32/8.
- The candidate was loaded from a pinned clean snapshot (bench/snap/6a31419); the live worktree had uncommitted edits.
