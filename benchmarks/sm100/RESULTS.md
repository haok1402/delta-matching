# Results: kda-sm100 candidates vs FA4 (hd128, GB200) — maintained by fa4-bench

All rows are measured by attnbench.py. The candidate and FA4 are timed in the same process on the same GPU, interleaved
in 6 rounds, with every GPU busy (4 shards). Speedup = FA4_ms / cand_ms, geomean over 12 shapes (H16/16 + H32/8,
N 1k-32k). "e2e" means the time starts from bf16 q/k/v, so the quantization producers are included, as in training.

> **Stale mode diverges in training** at sm100-mx precision (dm-validator, 2026-11-08; see validator/README.md). Its speed is reported for reference only; the headline is **matched, causal, fwd+bwd**.

| date | cand (sha) | pass | causal geomean (min-max) | non-causal geomean (min-max) | FA4 fp8 fwd same run | rel.err O vs fp32 | raw |
|---|---|---|---|---|---|---|---|
| 2026-11-08 17:38 | C1-v1 fwd e2e (811db9e) | fwd | 0.902 (0.624-1.170) | 0.936 (0.705-1.161) | 1.173 / 1.030 | 5.0-5.7e-2 | results/c1v1_fwd.s*.jsonl |
| 2026-11-08 17:53 | B1 fwd+bwd, stale mode, deterministic bwd (kda/cands/b1_cur.py @ wt/kda-sm100 7ed21b0) | fwd_bwd | 0.639 (0.368-1.003) vs FA4; 0.661 vs FA4-det | 0.728 (0.438-1.089); 0.746 vs FA4-det | - | O 5e-2, dQ/dK 6.7e-2, dV 5e-2 | results/b1_stale.s*.jsonl |
| 2026-11-08 18:23 | FULL matched (api.dm_attn_func, pinned snap 6a31419: C1-v2 fwd + P0 kernel + B1 det bwd) | fwd_bwd | **0.729** (0.537-0.891); 0.756 vs FA4-det | 0.771 (0.587-0.945); 0.793 vs FA4-det | - | O 5e-2, dQ/dK 6.5e-2, dV 5e-2 | results/full_6a31419.s*.jsonl |
| 2026-11-08 18:23 | FULL stale [diverges in training] (same build) | fwd_bwd | 0.686 (0.385-1.047); 0.711 vs FA4-det | 0.800 (0.462-1.169); 0.823 vs FA4-det | - | same | same |
| 2026-11-08 18:53 | FULL matched (snap 02016db: stale delta fused into dO quantizer) | fwd_bwd | **0.730** (0.538-0.893); 0.758 vs FA4-det | 0.772 (0.589-0.942); 0.796 vs FA4-det | - | no SUSPECT | results/full_02016db.s*.jsonl |
| 2026-11-08 18:53 | FULL stale [diverges in training] (snap 02016db) | fwd_bwd | 0.861 (0.592-1.097); 0.894 vs FA4-det | 0.945 (0.672-1.186); 0.975 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 19:16 | FULL matched (snap 5a912b2: P0-v2, deg-5 FMA ex2 polynomial in pass 0) | fwd_bwd | **0.729** (0.540-0.893); 0.760 vs FA4-det | 0.789; 0.809 vs FA4-det | - | no SUSPECT | results/full_5a912b2.s*.jsonl |
| 2026-11-08 19:16 | FULL stale [diverges in training] (snap 5a912b2) | fwd_bwd | 0.858; 0.894 vs FA4-det | 0.950; 0.973 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 19:40 | FULL matched (snap d227693: B1-v3, delta smem loads hoisted; numerics bitwise = 5a912b2) | fwd_bwd | **0.763** (+4.7% vs 5a912b2); 0.791 vs FA4-det | 0.802 (+1.7%); 0.826 vs FA4-det | - | no SUSPECT | results/full_d227693.s*.jsonl |
| 2026-11-08 19:40 | FULL stale [diverges in training] (snap d227693) | fwd_bwd | 0.906; 0.940 vs FA4-det | 0.971; 1.000 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 20:10 | **VALIDATED** FULL matched (snap 21b6b15, in validate/VALIDATED.md; = d227693 + debug hook) on nvl72053-T07 | fwd_bwd | **0.745** (0.551-0.931); 0.776 vs FA4-det | 0.796; 0.821 vs FA4-det | - | no SUSPECT | results/full_21b6b15.s*.jsonl |
| 2026-11-08 20:10 | FULL stale [diverges in training] (snap 21b6b15) | fwd_bwd | 0.884; 0.921 vs FA4-det | 0.966; 0.996 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 21:20 | VALIDATED FULL matched, snap 3122cc4 (= a911afe NaN fix + P0-v3 exp mix + OOB clamp), nvl72053-T07 GPUs 0,1,3 | fwd_bwd | **0.752** (0.551-0.94); 0.781 vs FA4-det | 0.805; 0.829 vs FA4-det | - | no SUSPECT | results/full_3122cc4g3.s*.jsonl |
| 2026-11-08 21:20 | FULL stale [diverges in training], snap 3122cc4 | fwd_bwd | 0.890; 0.924 vs FA4-det | 0.968; 0.996 vs FA4-det | - | no SUSPECT | same |
| 2026-11-08 21:55 | VALIDATED FULL matched, snap 30ec6cf (= 3122cc4 + aaca3a0 P0-v4 + 0c3f1e1), nvl72053-T07 GPUs 0-3, same lease as 3122cc4 | fwd_bwd | **0.757** (+1.0% vs 3122cc4 same lease); 0.789 vs FA4-det | 0.807; 0.832 vs FA4-det | - | no SUSPECT | results/ab_3122cc4_30ec6cf_30ec6cf.s*.jsonl |
| 2026-11-08 22:25 | **VALIDATED (current headline)** FULL matched, snap 9c690e1 (= 30ec6cf + e2e glue + bwd2; bitwise = 30ec6cf), nvl72053-T07 GPUs 1-3, same lease as 30ec6cf | fwd_bwd | **0.794** (0.581-0.961; +5.2% vs 30ec6cf same lease); 0.827 vs FA4-det | 0.850; 0.876 vs FA4-det | - | no SUSPECT | results/ab_30ec6cf_9c690e1_9c690e1.s*.jsonl |
| 2026-11-08 22:25 | FULL stale [diverges in training], snap 9c690e1 | fwd_bwd | 0.943; 0.982 vs FA4-det | 1.031; 1.063 vs FA4-det | - | no SUSPECT | same |

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

### Notes on FULL @02016db
- Stale improved from 0.686x to 0.861x causal fwd+bwd (0.800x to 0.945x non-causal). The stale bwd at 8k causal H16/16
  went from 2.97 to 2.41 ms, and 1k from 1.21 to 0.65 ms. Matched is unchanged at 0.730x (as expected: kernels unchanged).
- The matched-vs-stale gap is now the P0 (matched delta) pass: bwd 8k causal 3.07 vs 2.41 ms (+27%), 32k 10.98 vs
  8.44 ms (+30%). The stale bwd main kernel alone runs at 1141 TF (8k) and 1303 TF (32k) against FA4's 1304/1313 TF,
  so at long shapes the bwd main kernel is already at parity; **P0 is the main lever for the headline (matched).**

### Notes on FULL @5a912b2 vs 02016db
- Matched causal fwd+bwd 0.730x -> 0.729x (per-shape candidate time 1.000-1.005x faster: no change). Matched
  non-causal 0.772x -> 0.789x (+2.2%; the bwd is 1.2-2.7% faster on every shape). Stale and fwd unchanged (+-0.5%).
- The commit is mainly an accuracy change (more exact ex2 in pass 0). It does not make the causal matched pre-pass
  measurably faster: P0 still costs +27-29% of the bwd (8k causal: 3.06 vs 2.41 ms; 32k: 10.96 vs 8.44 ms).

### Delta d227693 vs 5a912b2 (python delta.py 5a912b2 d227693)
The bwd is 3-8% faster on every causal shape; the fwd is unchanged.

| arm | causal | pass | 5a912b2 | d227693 | change | d227693 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.731 | 0.748 | +2.3% | 0.783 | 1.012..1.033 | 12 |
| dm_matched | 0 | fwd | 0.939 | 0.937 | -0.2% | 0.937 | 1.000..1.004 | 12 |
| dm_matched | 0 | fwd_bwd | 0.789 | 0.802 | +1.7% | 0.826 | 1.010..1.027 | 12 |
| dm_matched | 1 | bwd | 0.682 | 0.721 | +5.7% | 0.762 | 1.031..1.084 | 12 |
| dm_matched | 1 | fwd | 0.899 | 0.899 | -0.0% | 0.900 | 1.000..1.001 | 12 |
| dm_matched | 1 | fwd_bwd | 0.729 | 0.763 | +4.7% | 0.791 | 1.024..1.068 | 12 |
| dm_stale | 0 | bwd | 0.929 | 0.958 | +3.1% | 1.003 | 1.014..1.044 | 12 |
| dm_stale | 0 | fwd | 0.939 | 0.936 | -0.2% | 0.936 | 0.998..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 0.950 | 0.971 | +2.3% | 1.000 | 1.011..1.038 | 12 |
| dm_stale | 1 | bwd | 0.838 | 0.898 | +7.2% | 0.950 | 1.038..1.110 | 12 |
| dm_stale | 1 | fwd | 0.899 | 0.898 | -0.1% | 0.899 | 0.997..1.001 | 12 |
| dm_stale | 1 | fwd_bwd | 0.858 | 0.906 | +5.6% | 0.940 | 1.027..1.084 | 12 |

### 21b6b15 (validated) and the node effect on the ratio
- Same node (nvl72053-T07), back to back: 21b6b15 vs d227693 = +0.1% matched causal fwd+bwd (no performance change, as
  expected for a debug-hook commit).
- Across nodes: d227693 measured 0.763x on nvl72063-T13 and 0.744x on nvl72053-T07. The DM kernel time is the same on
  both nodes (+-0.4%), but **FA4 causal is ~3% faster on nvl72053-T07** (fwd_bwd geomean 1.029x, up to 1.066x per
  shape); non-causal FA4 is the same on both. So a speedup ratio carries about +-2-3% node dependence. Rule from now on:
  commit-to-commit deltas only from runs on the same node, back to back (the previous commit is re-run if the node
  changed). Headline = the latest same-node number, with the node named.
- **Current validated headline (21b6b15): matched causal fwd+bwd 0.745x FA4 (0.776x vs FA4 deterministic) on
  nvl72053-T07; 0.763x on nvl72063-T13.**

| arm | causal | pass | d227693n2 | 21b6b15 | change | 21b6b15 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.747 | 0.746 | -0.1% | 0.783 | 0.991..1.003 | 12 |
| dm_matched | 0 | fwd | 0.939 | 0.927 | -1.3% | 0.925 | 0.966..1.006 | 12 |
| dm_matched | 0 | fwd_bwd | 0.801 | 0.796 | -0.6% | 0.821 | 0.986..1.002 | 12 |
| dm_matched | 1 | bwd | 0.702 | 0.701 | -0.1% | 0.748 | 1.000..1.002 | 12 |
| dm_matched | 1 | fwd | 0.877 | 0.878 | +0.1% | 0.879 | 0.997..1.001 | 12 |
| dm_matched | 1 | fwd_bwd | 0.744 | 0.745 | +0.1% | 0.776 | 0.999..1.001 | 12 |
| dm_stale | 0 | bwd | 0.955 | 0.959 | +0.4% | 1.006 | 1.000..1.009 | 12 |
| dm_stale | 0 | fwd | 0.939 | 0.927 | -1.3% | 0.925 | 0.966..1.006 | 12 |
| dm_stale | 0 | fwd_bwd | 0.969 | 0.966 | -0.3% | 0.996 | 0.992..1.003 | 12 |
| dm_stale | 1 | bwd | 0.875 | 0.874 | -0.1% | 0.933 | 0.999..1.002 | 12 |
| dm_stale | 1 | fwd | 0.877 | 0.879 | +0.2% | 0.880 | 0.999..1.000 | 12 |
| dm_stale | 1 | fwd_bwd | 0.883 | 0.884 | +0.1% | 0.921 | 0.999..1.001 | 12 |

> **2026-11-08 20:12 caveat (dm-validator):** 21b6b15 and all B1 builds give NaN dQ when N has an odd number of
> 128-blocks. The main-suite shapes (N = 1k..32k) are all even, so the timings above stand, but 21b6b15 is no longer a valid
> headline. Fix candidate: a911afe (validation running). The headline moves to a911afe once it is in validate/VALIDATED.md.

### a911afe (NaN-dQ fix + causal mask skip), pre-measured 2026-11-08 20:50, PENDING VALIDATION (not a headline yet)
Same node as 21b6b15 (nvl72053-T07), 3 GPUs: matched causal fwd+bwd 0.742x FA4 (0.774x vs det); no change vs 21b6b15
(per-shape candidate time 0.998-1.006x). The claimed +2% on causal MHA is not visible: the H16/16 causal bwd is
0.998-1.005x on every N. It becomes the headline once it is listed in validate/VALIDATED.md.

| arm | causal | pass | 21b6b15 | a911afe | change | a911afe vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.746 | 0.742 | -0.6% | 0.779 | 1.000..1.004 | 12 |
| dm_matched | 0 | fwd | 0.927 | 0.920 | -0.8% | 0.920 | 0.998..1.004 | 12 |
| dm_matched | 0 | fwd_bwd | 0.796 | 0.793 | -0.4% | 0.817 | 0.999..1.004 | 12 |
| dm_matched | 1 | bwd | 0.701 | 0.701 | -0.0% | 0.746 | 0.997..1.006 | 12 |
| dm_matched | 1 | fwd | 0.878 | 0.872 | -0.7% | 0.871 | 0.996..1.008 | 12 |
| dm_matched | 1 | fwd_bwd | 0.745 | 0.742 | -0.3% | 0.774 | 0.998..1.006 | 12 |
| dm_stale | 0 | bwd | 0.959 | 0.953 | -0.6% | 1.002 | 0.998..1.005 | 12 |
| dm_stale | 0 | fwd | 0.927 | 0.920 | -0.8% | 0.920 | 0.998..1.004 | 12 |
| dm_stale | 0 | fwd_bwd | 0.966 | 0.962 | -0.4% | 0.990 | 0.997..1.004 | 12 |
| dm_stale | 1 | bwd | 0.874 | 0.874 | -0.0% | 0.930 | 0.996..1.006 | 12 |
| dm_stale | 1 | fwd | 0.879 | 0.872 | -0.8% | 0.872 | 0.996..1.008 | 12 |
| dm_stale | 1 | fwd_bwd | 0.884 | 0.881 | -0.3% | 0.918 | 0.998..1.006 | 12 |

### 3122cc4 (validated; supersedes 21b6b15 / a911afe, which have OOB reads)
- **Headline: matched causal fwd+bwd 0.752x FA4 (0.781x vs FA4 deterministic), nvl72053-T07.** The same node gave 0.742x
  (a911afe) and 0.745x (21b6b15) earlier.
- Kernel performance has not changed since d227693: per-shape candidate time vs a911afe is 0.998-1.004x (fwd
  0.989-1.000x). P0-v3 (-3% on pass 0 alone) is about 0.5% of the matched bwd, below the noise.
- The ratio still moved +1.3% (3-GPU run) or +1.7% (1-GPU run on GPU0) because **FA4's time depends on which GPUs the
  shards land on** (a911afe on GPUs 1,2,3; 3122cc4 on 0,1,3 or 0 alone). The DM kernel's time is the same on every GPU.
  So treat headline ratios as +-1.5% (GPU placement) and +-2-3% (node). For commit-to-commit deltas, the trustworthy
  metric is the per-shape candidate-time column (same node). The ratio deltas above are within placement noise.

| arm | causal | pass | a911afe | 3122cc4g3 | change | 3122cc4g3 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.742 | 0.756 | +1.9% | 0.792 | 1.005..1.016 | 12 |
| dm_matched | 0 | fwd | 0.920 | 0.929 | +1.0% | 0.930 | 0.994..1.003 | 12 |
| dm_matched | 0 | fwd_bwd | 0.793 | 0.805 | +1.5% | 0.829 | 1.003..1.012 | 12 |
| dm_matched | 1 | bwd | 0.701 | 0.709 | +1.2% | 0.754 | 1.000..1.006 | 12 |
| dm_matched | 1 | fwd | 0.872 | 0.882 | +1.1% | 0.882 | 0.989..1.000 | 12 |
| dm_matched | 1 | fwd_bwd | 0.742 | 0.752 | +1.3% | 0.781 | 0.998..1.004 | 12 |
| dm_stale | 0 | bwd | 0.953 | 0.960 | +0.7% | 1.007 | 0.996..1.000 | 12 |
| dm_stale | 0 | fwd | 0.920 | 0.929 | +1.0% | 0.930 | 0.996..1.001 | 12 |
| dm_stale | 0 | fwd_bwd | 0.962 | 0.968 | +0.6% | 0.996 | 0.996..1.004 | 12 |
| dm_stale | 1 | bwd | 0.874 | 0.881 | +0.8% | 0.936 | 0.996..1.001 | 12 |
| dm_stale | 1 | fwd | 0.872 | 0.882 | +1.1% | 0.882 | 0.992..1.002 | 12 |
| dm_stale | 1 | fwd_bwd | 0.881 | 0.890 | +1.0% | 0.924 | 0.992..1.000 | 12 |

### aaca3a0 (P0-v4, branch kda-sm100-p0r3) vs 3122cc4: same-lease A/B/A, PENDING VALIDATION (2026-11-08 21:50)
New method (bench/run_ab.sh): A, B, then A again in ONE lease, shard i always on the same GPU (nvl72053-T07, GPUs 0,1,2),
so placement and node are identical. A vs A-repeat gives the noise floor: per-shape candidate time 0.993-1.005.
- matched causal bwd: **+0.3..+1.0% faster on every shape** (above the noise floor, below the expected ~2%). For
  example H16/16 8k 2.894 -> 2.868 ms, 32k 10.314 -> 10.215 ms. The fwd+bwd ratio is 0.752x -> 0.756x (+0.5%).
- non-causal fwd 0.999..1.036 (+2% geomean): unexpected for a pass-0-only change. Worth checking whether the fwd shares
  the changed exp code. Matched non-causal bwd and stale are unchanged.

B vs A:
| arm | causal | pass | 3122cc4 | aaca3a0 | change | aaca3a0 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.757 | 0.757 | -0.0% | 0.794 | 0.995..1.008 | 12 |
| dm_matched | 0 | fwd | 0.927 | 0.945 | +2.0% | 0.943 | 0.999..1.036 | 12 |
| dm_matched | 0 | fwd_bwd | 0.806 | 0.811 | +0.6% | 0.836 | 0.999..1.013 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.716 | +0.7% | 0.761 | 1.003..1.010 | 12 |
| dm_matched | 1 | fwd | 0.879 | 0.880 | +0.1% | 0.881 | 1.000..1.004 | 12 |
| dm_matched | 1 | fwd_bwd | 0.752 | 0.756 | +0.5% | 0.787 | 1.003..1.008 | 12 |
| dm_stale | 0 | bwd | 0.960 | 0.958 | -0.2% | 1.006 | 0.997..1.006 | 12 |
| dm_stale | 0 | fwd | 0.926 | 0.944 | +2.0% | 0.943 | 1.000..1.036 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.973 | +0.5% | 1.003 | 0.996..1.010 | 12 |
| dm_stale | 1 | bwd | 0.883 | 0.882 | -0.0% | 0.938 | 0.997..1.002 | 12 |
| dm_stale | 1 | fwd | 0.879 | 0.880 | +0.1% | 0.881 | 0.999..1.004 | 12 |
| dm_stale | 1 | fwd_bwd | 0.890 | 0.889 | -0.1% | 0.926 | 0.998..1.001 | 12 |

A vs A-repeat (noise floor):
| arm | causal | pass | 3122cc4 | 3122cc4_r2 | change | 3122cc4_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.757 | 0.759 | +0.3% | 0.794 | 0.996..1.005 | 12 |
| dm_matched | 0 | fwd | 0.927 | 0.928 | +0.2% | 0.928 | 0.995..1.002 | 12 |
| dm_matched | 0 | fwd_bwd | 0.806 | 0.809 | +0.4% | 0.832 | 0.993..1.004 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.712 | +0.2% | 0.753 | 0.994..1.001 | 12 |
| dm_matched | 1 | fwd | 0.879 | 0.880 | +0.2% | 0.881 | 0.996..1.002 | 12 |
| dm_matched | 1 | fwd_bwd | 0.752 | 0.753 | +0.2% | 0.783 | 0.997..1.000 | 12 |
| dm_stale | 0 | bwd | 0.960 | 0.964 | +0.4% | 1.009 | 0.998..1.012 | 12 |
| dm_stale | 0 | fwd | 0.926 | 0.928 | +0.2% | 0.927 | 0.997..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.972 | +0.4% | 1.000 | 0.992..1.003 | 12 |
| dm_stale | 1 | bwd | 0.883 | 0.884 | +0.1% | 0.936 | 0.991..1.000 | 12 |
| dm_stale | 1 | fwd | 0.879 | 0.880 | +0.1% | 0.880 | 0.992..1.001 | 12 |
| dm_stale | 1 | fwd_bwd | 0.890 | 0.892 | +0.2% | 0.927 | 0.995..1.001 | 12 |

### 30ec6cf (validated; superseded by 9c690e1) vs 3122cc4, same lease, GPUs 0-3 of nvl72053-T07 (2026-11-08 21:55)
**matched causal fwd+bwd 0.757x FA4 (0.789x vs FA4 deterministic)**, +1.0% vs 3122cc4. Per-shape candidate time is
1.005-1.019x, above the 0.993-1.005 noise floor, from both bwd (+0.2..1.0%) and fwd (+0.2..5.0%, causal). The A-repeat
was cancelled to free the GPUs for kda-sm100; the noise floor is from the 3122cc4/aaca3a0 A/B/A run.

| arm | causal | pass | 3122cc4 | 30ec6cf | change | 30ec6cf vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.756 | 0.757 | +0.1% | 0.794 | 0.996..1.001 | 12 |
| dm_matched | 0 | fwd | 0.928 | 0.941 | +1.4% | 0.940 | 0.999..1.041 | 12 |
| dm_matched | 0 | fwd_bwd | 0.806 | 0.807 | +0.2% | 0.832 | 0.995..1.012 | 12 |
| dm_matched | 1 | bwd | 0.708 | 0.713 | +0.7% | 0.759 | 1.002..1.010 | 12 |
| dm_matched | 1 | fwd | 0.878 | 0.893 | +1.8% | 0.895 | 1.002..1.050 | 12 |
| dm_matched | 1 | fwd_bwd | 0.749 | 0.757 | +1.0% | 0.789 | 1.005..1.019 | 12 |
| dm_stale | 0 | bwd | 0.961 | 0.962 | +0.2% | 1.010 | 0.992..1.007 | 12 |
| dm_stale | 0 | fwd | 0.928 | 0.941 | +1.5% | 0.940 | 0.999..1.041 | 12 |
| dm_stale | 0 | fwd_bwd | 0.969 | 0.971 | +0.2% | 1.001 | 0.997..1.013 | 12 |
| dm_stale | 1 | bwd | 0.879 | 0.880 | +0.0% | 0.936 | 1.000..1.001 | 12 |
| dm_stale | 1 | fwd | 0.878 | 0.894 | +1.9% | 0.896 | 1.002..1.057 | 12 |
| dm_stale | 1 | fwd_bwd | 0.887 | 0.892 | +0.5% | 0.930 | 1.000..1.016 | 12 |

### f5a04c6 (e2e glue, merge of 50ded49) vs 30ec6cf, same-lease A/B/A, GPUs 1-3 nvl72053-T07, PENDING VALIDATION (22:11)
Noise floor (A vs A-repeat): per-shape 0.994-1.004.
- **matched causal fwd+bwd 0.753x -> 0.776x FA4 (+3.0%), 0.807x vs FA4-det.** Every shape is faster (1.013-1.057).
- matched causal bwd per N: H16/16 1k..32k +7.8/+5.7/+4.6/+4.0/+3.4/+3.1%; H32/8 +5.2/+3.8/+2.9/+2.1/+1.7/+1.6%.
  This confirms kda's fork screening (bwd +3.4-5% MHA, +1.6-2.9% GQA). The fwd is unchanged.
- matched non-causal fwd+bwd +1.7%. Stale +1.9..3.3%. No SUSPECT flags.
- Becomes the headline when it is in validate/VALIDATED.md (I will reuse this run if the node is unchanged).

B vs A:
| arm | causal | pass | 30ec6cf | f5a04c6 | change | f5a04c6 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.770 | +2.4% | 0.810 | 1.001..1.063 | 12 |
| dm_matched | 0 | fwd | 0.936 | 0.933 | -0.4% | 0.934 | 0.995..1.003 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.819 | +1.7% | 0.845 | 1.000..1.046 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.736 | +3.5% | 0.783 | 1.016..1.078 | 12 |
| dm_matched | 1 | fwd | 0.889 | 0.886 | -0.3% | 0.889 | 0.996..1.001 | 12 |
| dm_matched | 1 | fwd_bwd | 0.753 | 0.776 | +3.0% | 0.807 | 1.013..1.057 | 12 |
| dm_stale | 0 | bwd | 0.955 | 0.983 | +2.9% | 1.033 | 0.999..1.072 | 12 |
| dm_stale | 0 | fwd | 0.936 | 0.933 | -0.4% | 0.934 | 0.995..1.002 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.987 | +1.9% | 1.017 | 0.998..1.050 | 12 |
| dm_stale | 1 | bwd | 0.876 | 0.913 | +4.2% | 0.971 | 1.019..1.089 | 12 |
| dm_stale | 1 | fwd | 0.889 | 0.887 | -0.3% | 0.889 | 0.996..1.000 | 12 |
| dm_stale | 1 | fwd_bwd | 0.887 | 0.917 | +3.3% | 0.953 | 1.014..1.063 | 12 |

A vs A-repeat:
| arm | causal | pass | 30ec6cf | 30ec6cf_r2 | change | 30ec6cf_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.754 | +0.2% | 0.791 | 0.998..1.004 | 12 |
| dm_matched | 0 | fwd | 0.936 | 0.938 | +0.1% | 0.937 | 1.000..1.003 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.804 | -0.2% | 0.830 | 0.994..1.002 | 12 |
| dm_matched | 1 | bwd | 0.711 | 0.712 | +0.1% | 0.755 | 0.999..1.001 | 12 |
| dm_matched | 1 | fwd | 0.889 | 0.889 | -0.1% | 0.889 | 1.000..1.003 | 12 |
| dm_matched | 1 | fwd_bwd | 0.753 | 0.754 | +0.1% | 0.785 | 1.000..1.002 | 12 |
| dm_stale | 0 | bwd | 0.955 | 0.957 | +0.3% | 1.004 | 0.995..1.008 | 12 |
| dm_stale | 0 | fwd | 0.936 | 0.937 | +0.1% | 0.936 | 0.999..1.004 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.967 | -0.1% | 0.999 | 0.998..1.003 | 12 |
| dm_stale | 1 | bwd | 0.876 | 0.877 | +0.1% | 0.931 | 0.999..1.002 | 12 |
| dm_stale | 1 | fwd | 0.889 | 0.889 | -0.1% | 0.889 | 0.998..1.005 | 12 |
| dm_stale | 1 | fwd_bwd | 0.887 | 0.888 | +0.1% | 0.924 | 0.999..1.001 | 12 |

### 9c690e1 (merge of f5a04c6 e2e glue + d3a5837 bwd2) vs 30ec6cf, same-lease A/B/A, GPUs 1-3 nvl72053-T07, VALIDATED 22:16 by dm-validator = CURRENT HEADLINE (22:22)
Noise floor (A vs A-repeat): per-shape 0.993-1.007.
- **matched causal fwd+bwd 0.755x -> 0.794x FA4 (+5.2%), 0.827x vs FA4-det.** Every shape is faster (1.034-1.068).
- matched causal bwd per N: H16/16 +9.5/+8.9/+8.8/+8.1/+8.5/+8.4% (1k..32k); H32/8 +5.8/+5.0/+4.6/+4.9/+4.3/+4.4%. fwd unchanged.
- matched causal fwd+bwd vs FA4 per N: H16/16 0.795/0.798/0.790/0.820/0.849/0.931; H32/8 0.581/0.642/0.709/0.822/0.924/0.961.
  Short GQA (1k-2k) is the weakest point.
- matched non-causal 0.850x (+5.6%). Stale [diverges in training] now beats FA4 on non-causal (1.031x fwd+bwd, bwd 1.044x).
- Sanity errors vs fp32 are identical to 30ec6cf (O 5.0-5.7e-2, dQ/dK 6.4-6.7e-2, dV 4.9-5.5e-2): the speed-up does not
  come from changed numerics. No SUSPECT flags.
- This becomes the headline once it is in validate/VALIDATED.md (same node: this run is reused).

B vs A:
| arm | causal | pass | 30ec6cf | 9c690e1 | change | 9c690e1 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.807 | +7.4% | 0.850 | 1.046..1.106 | 12 |
| dm_matched | 0 | fwd | 0.937 | 0.935 | -0.2% | 0.934 | 0.987..1.007 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.850 | +5.6% | 0.876 | 1.027..1.079 | 12 |
| dm_matched | 1 | bwd | 0.712 | 0.759 | +6.6% | 0.807 | 1.043..1.095 | 12 |
| dm_matched | 1 | fwd | 0.887 | 0.887 | +0.1% | 0.887 | 0.992..1.007 | 12 |
| dm_matched | 1 | fwd_bwd | 0.755 | 0.794 | +5.2% | 0.827 | 1.034..1.068 | 12 |
| dm_stale | 0 | bwd | 0.956 | 1.044 | +9.3% | 1.100 | 1.054..1.139 | 12 |
| dm_stale | 0 | fwd | 0.937 | 0.936 | -0.1% | 0.934 | 0.987..1.006 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 1.031 | +6.5% | 1.063 | 1.033..1.099 | 12 |
| dm_stale | 1 | bwd | 0.878 | 0.949 | +8.1% | 1.009 | 1.053..1.112 | 12 |
| dm_stale | 1 | fwd | 0.887 | 0.888 | +0.0% | 0.887 | 0.992..1.006 | 12 |
| dm_stale | 1 | fwd_bwd | 0.889 | 0.943 | +6.0% | 0.982 | 1.041..1.082 | 12 |

A vs A-repeat:
| arm | causal | pass | 30ec6cf | 30ec6cf_r2 | change | 30ec6cf_r2 vs FA4-det | per-shape speedup of cand (min..max) | n |
|---|---|---|---|---|---|---|---|---|
| dm_matched | 0 | bwd | 0.752 | 0.751 | -0.1% | 0.791 | 0.999..1.006 | 12 |
| dm_matched | 0 | fwd | 0.937 | 0.932 | -0.5% | 0.935 | 0.996..1.007 | 12 |
| dm_matched | 0 | fwd_bwd | 0.805 | 0.803 | -0.2% | 0.829 | 0.993..1.004 | 12 |
| dm_matched | 1 | bwd | 0.712 | 0.711 | -0.2% | 0.756 | 0.999..1.005 | 12 |
| dm_matched | 1 | fwd | 0.887 | 0.887 | +0.0% | 0.888 | 0.996..1.007 | 12 |
| dm_matched | 1 | fwd_bwd | 0.755 | 0.755 | +0.0% | 0.784 | 1.000..1.005 | 12 |
| dm_stale | 0 | bwd | 0.956 | 0.953 | -0.2% | 1.004 | 0.995..1.001 | 12 |
| dm_stale | 0 | fwd | 0.937 | 0.932 | -0.5% | 0.935 | 0.996..1.006 | 12 |
| dm_stale | 0 | fwd_bwd | 0.968 | 0.965 | -0.3% | 0.997 | 0.997..1.001 | 12 |
| dm_stale | 1 | bwd | 0.878 | 0.876 | -0.2% | 0.931 | 0.999..1.004 | 12 |
| dm_stale | 1 | fwd | 0.887 | 0.888 | +0.0% | 0.889 | 0.998..1.007 | 12 |
| dm_stale | 1 | fwd_bwd | 0.889 | 0.889 | -0.0% | 0.923 | 0.999..1.005 | 12 |
