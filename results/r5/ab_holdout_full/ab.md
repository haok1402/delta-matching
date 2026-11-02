# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r3_do2 (library sha256 7d491ed1a5560e4e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/h1b (library sha256 bc2121b089935ca3)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.289 | 17.229 | -0.35% | 0.89% | 0.17% |
| prod_fwd | 1.781 | 1.777 | -0.18% | 0.28% | 0.09% |
| prod_bwd | 1.199 | 1.199 | -0.01% | 0.17% | 0.14% |
| pass0 | 3.383 | 3.361 | -0.64% | 1.00% | 0.22% |
| pass1 | 45.605 | 44.967 | -1.40% | 0.94% | 0.15% |
| pass2 | 10.565 | 10.546 | -0.18% | 0.40% | 0.04% |
| twin | 0.804 | 0.804 | +0.12% | 0.08% | 0.17% |
| glue | 5.479 | 5.470 | -0.15% | 0.28% | 0.01% |
| e2e_fwd | 22.523 | 22.505 | -0.08% | 0.55% | 0.26% |
| e2e_fwdbwd | 86.127 | 85.719 | -0.47% | 0.81% | 0.41% |

### Per-kernel sums, affected subset D128

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 8.032 | 8.025 | -0.09% | 0.84% | 0.34% |
| prod_fwd | 0.968 | 0.968 | -0.03% | 0.22% | 0.15% |
| prod_bwd | 0.537 | 0.538 | +0.17% | 0.23% | 0.20% |
| pass0 | 1.748 | 1.738 | -0.54% | 1.01% | 0.39% |
| pass1 | 20.626 | 20.136 | -2.38% | 0.82% | 0.23% |
| pass2 | 5.037 | 5.034 | -0.05% | 0.14% | 0.10% |
| twin | 0.370 | 0.369 | -0.06% | 0.27% | 0.39% |
| glue | 2.774 | 2.774 | -0.01% | 0.16% | 0.06% |
| e2e_fwd | 10.864 | 10.860 | -0.03% | 0.50% | 0.91% |
| e2e_fwdbwd | 40.102 | 39.747 | -0.88% | 0.76% | 0.30% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 6.974 | 6.945 | -0.40% | 2.67% | 0.82% |
| D128 B1 H24/8 N12288 | causal | stale | 5.935 | 5.901 | -0.57% | 0.82% | 0.61% |
| D128 B1 H24/8 N12288 | full | stale | 11.228 | 11.079 | -1.32% | 1.12% | 0.72% |
| D128 B2 H32/32 N3072 | causal | matched | 1.693 | 1.688 | -0.35% | 1.29% | 0.77% |
| D128 B2 H32/32 N3072 | causal | stale | 1.495 | 1.492 | -0.19% | 1.22% | 0.62% |
| D128 B2 H32/32 N3072 | full | stale | 2.345 | 2.310 | -1.50% | 0.13% | 0.37% |
| D128 B3 H12/4 N6144 | causal | matched | 3.074 | 3.062 | -0.39% | 1.43% | 1.24% |
| D128 B3 H12/4 N6144 | causal | stale | 2.650 | 2.639 | -0.41% | 0.68% | 0.10% |
| D128 B3 H12/4 N6144 | full | stale | 4.709 | 4.630 | -1.66% | 0.59% | 0.53% |
| D256 B1 H12/4 N12288 | causal | matched | 6.786 | 6.839 | +0.77% | 2.22% | 1.81% |
| D256 B1 H12/4 N12288 | causal | stale | 5.940 | 5.926 | -0.24% | 2.20% | 0.71% |
| D256 B1 H12/4 N12288 | full | stale | 11.141 | 11.090 | -0.45% | 1.05% | 0.89% |
| D256 B2 H16/1 N5120 | causal | matched | 3.402 | 3.408 | +0.17% | 0.06% | 0.83% |
| D256 B2 H16/1 N5120 | causal | stale | 3.026 | 3.006 | -0.67% | 1.22% | 0.06% |
| D256 B2 H16/1 N5120 | full | stale | 5.436 | 5.430 | -0.13% | 1.35% | 0.17% |
| D256 B3 H6/2 N6144 | causal | matched | 2.980 | 2.960 | -0.64% | 1.33% | 0.72% |
| D256 B3 H6/2 N6144 | causal | stale | 2.634 | 2.634 | +0.00% | 1.36% | 0.21% |
| D256 B3 H6/2 N6144 | full | stale | 4.680 | 4.680 | -0.01% | 1.34% | 0.12% |

Worst single-shape fwd+bwd change (B vs A): +0.77%
