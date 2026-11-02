# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r3_do2 (library sha256 7d491ed1a5560e4e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/gf_pre (library sha256 31482f56742b8cfe)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.168 | 17.133 | -0.21% | 0.24% | 0.22% |
| prod_fwd | 1.776 | 1.776 | -0.04% | 0.09% | 0.12% |
| prod_bwd | 1.197 | 1.197 | -0.02% | 0.06% | 0.16% |
| pass0 | 3.341 | 3.336 | -0.14% | 0.37% | 0.17% |
| pass1 | 45.360 | 45.310 | -0.11% | 0.25% | 0.22% |
| pass2 | 10.528 | 10.526 | -0.02% | 0.05% | 0.08% |
| twin | 0.806 | 0.806 | -0.03% | 0.04% | 0.07% |
| glue | 5.456 | 5.461 | +0.10% | 0.08% | 0.09% |
| e2e_fwd | 22.547 | 22.485 | -0.27% | 0.08% | 0.19% |
| e2e_fwdbwd | 86.113 | 85.932 | -0.21% | 0.11% | 0.27% |

### Per-kernel sums, affected subset D256

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 9.204 | 8.982 | -2.41% | 0.09% | 0.07% |
| prod_fwd | 0.810 | 0.810 | -0.07% | 0.09% | 0.30% |
| prod_bwd | 0.661 | 0.661 | +0.04% | 0.10% | 0.11% |
| pass0 | 1.618 | 1.618 | -0.04% | 0.53% | 0.22% |
| pass1 | 24.882 | 24.862 | -0.08% | 0.09% | 0.07% |
| pass2 | 5.502 | 5.500 | -0.03% | 0.08% | 0.13% |
| twin | 0.436 | 0.436 | +0.02% | 0.05% | 0.24% |
| glue | 2.692 | 2.695 | +0.14% | 0.09% | 0.07% |
| e2e_fwd | 11.683 | 11.437 | -2.10% | 0.19% | 0.32% |
| e2e_fwdbwd | 46.057 | 45.786 | -0.59% | 0.08% | 0.29% |

### Shapes over the 1.0% fwd+bwd gate

- (128, 1, 24, 8, 12288, 'causal', 'stale'): +1.42% (A reps [5.935, 5.964, 5.937], B reps [6.024, 5.972, 6.022]; spreads 0.48% / 0.86%) (within spread)
- (128, 2, 32, 32, 3072, 'causal', 'matched'): +1.18% (A reps [1.701, 1.695, 1.698], B reps [1.718, 1.722, 1.703]; spreads 0.40% / 1.10%) (within spread)
- (128, 2, 32, 32, 3072, 'causal', 'stale'): +1.39% (A reps [1.501, 1.497, 1.495], B reps [1.52, 1.518, 1.509]; spreads 0.38% / 0.74%) (within spread)
- (128, 3, 12, 4, 6144, 'causal', 'stale'): +1.21% (A reps [2.635, 2.635, 2.639], B reps [2.667, 2.668, 2.663]; spreads 0.15% / 0.19%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 6.968 | 7.008 | +0.57% | 0.78% | 0.58% |
| D128 B1 H24/8 N12288 | causal | stale | 5.937 | 6.022 | +1.42% | 0.48% | 0.86% |
| D128 B1 H24/8 N12288 | full | stale | 11.197 | 11.084 | -1.01% | 1.17% | 1.04% |
| D128 B2 H32/32 N3072 | causal | matched | 1.698 | 1.718 | +1.18% | 0.40% | 1.10% |
| D128 B2 H32/32 N3072 | causal | stale | 1.497 | 1.518 | +1.39% | 0.38% | 0.74% |
| D128 B2 H32/32 N3072 | full | stale | 2.349 | 2.342 | -0.30% | 0.35% | 0.09% |
| D128 B3 H12/4 N6144 | causal | matched | 3.089 | 3.105 | +0.50% | 0.88% | 0.78% |
| D128 B3 H12/4 N6144 | causal | stale | 2.635 | 2.667 | +1.21% | 0.15% | 0.19% |
| D128 B3 H12/4 N6144 | full | stale | 4.687 | 4.683 | -0.07% | 0.38% | 0.11% |
| D256 B1 H12/4 N12288 | causal | matched | 6.792 | 6.768 | -0.36% | 0.91% | 1.33% |
| D256 B1 H12/4 N12288 | causal | stale | 5.924 | 5.878 | -0.77% | 0.06% | 0.65% |
| D256 B1 H12/4 N12288 | full | stale | 11.117 | 11.083 | -0.31% | 0.04% | 0.04% |
| D256 B2 H16/1 N5120 | causal | matched | 3.411 | 3.381 | -0.88% | 0.53% | 0.31% |
| D256 B2 H16/1 N5120 | causal | stale | 3.025 | 2.989 | -1.19% | 0.64% | 0.79% |
| D256 B2 H16/1 N5120 | full | stale | 5.465 | 5.415 | -0.91% | 0.96% | 0.81% |
| D256 B3 H6/2 N6144 | causal | matched | 2.975 | 2.957 | -0.59% | 0.78% | 0.18% |
| D256 B3 H6/2 N6144 | causal | stale | 2.645 | 2.631 | -0.51% | 0.13% | 0.09% |
| D256 B3 H6/2 N6144 | full | stale | 4.702 | 4.683 | -0.40% | 0.15% | 0.47% |

Worst single-shape fwd+bwd change (B vs A): +1.42%
