# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r3_do2 (library sha256 7d491ed1a5560e4e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/pp (library sha256 4367ca998298e188)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.145 | 17.145 | +0.00% | 0.24% | 0.15% |
| prod_fwd | 1.776 | 1.775 | -0.07% | 0.04% | 0.15% |
| prod_bwd | 1.197 | 1.197 | -0.00% | 0.08% | 0.04% |
| pass0 | 3.329 | 3.329 | -0.01% | 0.33% | 0.12% |
| pass1 | 45.269 | 45.610 | +0.75% | 0.20% | 0.18% |
| pass2 | 10.527 | 10.482 | -0.43% | 0.06% | 0.12% |
| twin | 0.805 | 0.805 | -0.07% | 0.16% | 0.18% |
| glue | 5.463 | 5.463 | +0.00% | 0.09% | 0.08% |
| e2e_fwd | 22.484 | 22.476 | -0.04% | 0.03% | 0.14% |
| e2e_fwdbwd | 85.842 | 86.021 | +0.21% | 0.12% | 0.08% |

### Per-kernel sums, affected subset D128

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 7.947 | 7.953 | +0.07% | 0.34% | 0.17% |
| prod_fwd | 0.966 | 0.965 | -0.06% | 0.21% | 0.15% |
| prod_bwd | 0.537 | 0.536 | -0.06% | 0.27% | 0.06% |
| pass0 | 1.723 | 1.721 | -0.10% | 0.33% | 0.20% |
| pass1 | 20.416 | 20.768 | +1.72% | 0.37% | 0.10% |
| pass2 | 5.023 | 4.978 | -0.90% | 0.06% | 0.13% |
| twin | 0.370 | 0.370 | -0.12% | 0.61% | 0.32% |
| glue | 2.767 | 2.766 | -0.03% | 0.09% | 0.07% |
| e2e_fwd | 10.833 | 10.820 | -0.12% | 0.15% | 0.19% |
| e2e_fwdbwd | 39.911 | 40.211 | +0.75% | 0.49% | 0.17% |

### Shapes over the 1.0% fwd+bwd gate

- (128, 1, 24, 8, 12288, 'causal', 'matched'): +1.70% (A reps [6.968, 6.921, 6.916], B reps [6.991, 7.052, 7.039]; spreads 0.76% / 0.87%) (within spread)
- (128, 1, 24, 8, 12288, 'causal', 'stale'): +1.43% (A reps [5.935, 5.926, 5.926], B reps [6.007, 6.011, 6.021]; spreads 0.16% / 0.23%) BEYOND SPREAD
- (128, 3, 12, 4, 6144, 'causal', 'stale'): +1.07% (A reps [2.631, 2.636, 2.633], B reps [2.665, 2.662, 2.661]; spreads 0.19% / 0.13%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 6.921 | 7.039 | +1.70% | 0.76% | 0.87% |
| D128 B1 H24/8 N12288 | causal | stale | 5.926 | 6.011 | +1.43% | 0.16% | 0.23% |
| D128 B1 H24/8 N12288 | full | stale | 11.110 | 11.141 | +0.28% | 0.82% | 0.07% |
| D128 B2 H32/32 N3072 | causal | matched | 1.702 | 1.710 | +0.46% | 0.22% | 0.57% |
| D128 B2 H32/32 N3072 | causal | stale | 1.497 | 1.506 | +0.62% | 0.45% | 0.05% |
| D128 B2 H32/32 N3072 | full | stale | 2.345 | 2.349 | +0.16% | 0.20% | 0.15% |
| D128 B3 H12/4 N6144 | causal | matched | 3.093 | 3.098 | +0.16% | 1.75% | 0.20% |
| D128 B3 H12/4 N6144 | causal | stale | 2.633 | 2.662 | +1.07% | 0.19% | 0.13% |
| D128 B3 H12/4 N6144 | full | stale | 4.683 | 4.695 | +0.26% | 0.28% | 0.12% |
| D256 B1 H12/4 N12288 | causal | matched | 6.769 | 6.737 | -0.47% | 1.50% | 1.16% |
| D256 B1 H12/4 N12288 | causal | stale | 5.926 | 5.929 | +0.05% | 0.18% | 0.10% |
| D256 B1 H12/4 N12288 | full | stale | 11.123 | 11.027 | -0.87% | 0.08% | 0.78% |
| D256 B2 H16/1 N5120 | causal | matched | 3.391 | 3.391 | -0.01% | 0.06% | 0.20% |
| D256 B2 H16/1 N5120 | causal | stale | 3.025 | 3.023 | -0.06% | 0.82% | 0.46% |
| D256 B2 H16/1 N5120 | full | stale | 5.436 | 5.430 | -0.11% | 0.91% | 0.29% |
| D256 B3 H6/2 N6144 | causal | matched | 2.953 | 2.970 | +0.59% | 0.67% | 0.12% |
| D256 B3 H6/2 N6144 | causal | stale | 2.631 | 2.629 | -0.08% | 0.08% | 0.10% |
| D256 B3 H6/2 N6144 | full | stale | 4.677 | 4.674 | -0.05% | 0.12% | 0.08% |

Worst single-shape fwd+bwd change (B vs A): +1.70%
