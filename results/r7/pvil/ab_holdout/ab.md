# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r3_do2 (library sha256 7d491ed1a5560e4e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/pvil (library sha256 2b68543a6dc30633)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.175 | 17.001 | -1.01% | 0.24% | 0.04% |
| prod_fwd | 1.778 | 1.780 | +0.10% | 0.21% | 0.05% |
| prod_bwd | 1.199 | 1.202 | +0.20% | 0.07% | 0.05% |
| pass0 | 3.344 | 3.358 | +0.41% | 0.19% | 0.35% |
| pass1 | 45.330 | 45.418 | +0.19% | 0.21% | 0.01% |
| pass2 | 10.547 | 10.553 | +0.05% | 0.16% | 0.10% |
| twin | 0.806 | 0.805 | -0.04% | 0.08% | 0.21% |
| glue | 5.482 | 5.490 | +0.15% | 0.03% | 0.07% |
| e2e_fwd | 22.525 | 22.347 | -0.79% | 0.10% | 0.19% |
| e2e_fwdbwd | 86.108 | 85.962 | -0.17% | 0.14% | 0.10% |

### Per-kernel sums, affected subset D256

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 9.207 | 8.970 | -2.57% | 0.43% | 0.10% |
| prod_fwd | 0.811 | 0.811 | +0.03% | 0.08% | 0.07% |
| prod_bwd | 0.662 | 0.663 | +0.24% | 0.07% | 0.08% |
| pass0 | 1.620 | 1.624 | +0.27% | 0.54% | 0.50% |
| pass1 | 24.879 | 24.920 | +0.17% | 0.41% | 0.07% |
| pass2 | 5.518 | 5.520 | +0.04% | 0.26% | 0.13% |
| twin | 0.436 | 0.436 | -0.14% | 0.23% | 0.25% |
| glue | 2.706 | 2.710 | +0.15% | 0.09% | 0.13% |
| e2e_fwd | 11.662 | 11.426 | -2.02% | 0.10% | 0.29% |
| e2e_fwdbwd | 46.095 | 45.849 | -0.53% | 0.16% | 0.22% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 6.982 | 6.948 | -0.49% | 1.09% | 1.04% |
| D128 B1 H24/8 N12288 | causal | stale | 5.936 | 5.917 | -0.32% | 0.12% | 0.07% |
| D128 B1 H24/8 N12288 | full | stale | 11.174 | 11.244 | +0.63% | 0.00% | 0.11% |
| D128 B2 H32/32 N3072 | causal | matched | 1.691 | 1.703 | +0.75% | 0.89% | 0.04% |
| D128 B2 H32/32 N3072 | causal | stale | 1.497 | 1.507 | +0.71% | 0.66% | 0.36% |
| D128 B2 H32/32 N3072 | full | stale | 2.347 | 2.366 | +0.83% | 0.28% | 0.11% |
| D128 B3 H12/4 N6144 | causal | matched | 3.073 | 3.073 | -0.03% | 0.36% | 0.21% |
| D128 B3 H12/4 N6144 | causal | stale | 2.639 | 2.634 | -0.21% | 0.19% | 0.24% |
| D128 B3 H12/4 N6144 | full | stale | 4.675 | 4.721 | +0.98% | 0.26% | 0.47% |
| D256 B1 H12/4 N12288 | causal | matched | 6.849 | 6.807 | -0.62% | 0.17% | 0.72% |
| D256 B1 H12/4 N12288 | causal | stale | 5.931 | 5.920 | -0.18% | 0.63% | 0.56% |
| D256 B1 H12/4 N12288 | full | stale | 11.164 | 11.038 | -1.12% | 0.61% | 0.33% |
| D256 B2 H16/1 N5120 | causal | matched | 3.394 | 3.394 | -0.02% | 0.46% | 0.38% |
| D256 B2 H16/1 N5120 | causal | stale | 3.006 | 3.023 | +0.56% | 1.07% | 0.23% |
| D256 B2 H16/1 N5120 | full | stale | 5.446 | 5.421 | -0.46% | 0.79% | 0.10% |
| D256 B3 H6/2 N6144 | causal | matched | 2.979 | 2.975 | -0.11% | 0.29% | 0.98% |
| D256 B3 H6/2 N6144 | causal | stale | 2.644 | 2.629 | -0.57% | 0.82% | 0.89% |
| D256 B3 H6/2 N6144 | full | stale | 4.682 | 4.643 | -0.85% | 0.10% | 0.29% |

Worst single-shape fwd+bwd change (B vs A): +0.98%
