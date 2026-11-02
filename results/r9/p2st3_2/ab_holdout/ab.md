# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r3_do2 (library sha256 7d491ed1a5560e4e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p2st3 (library sha256 d2eee98774ebac50)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.304 | 17.349 | +0.26% | 0.35% | 0.31% |
| prod_fwd | 1.783 | 1.785 | +0.09% | 0.20% | 0.10% |
| prod_bwd | 1.201 | 1.201 | -0.01% | 0.05% | 0.08% |
| pass0 | 3.389 | 3.394 | +0.17% | 0.37% | 0.21% |
| pass1 | 45.665 | 45.791 | +0.28% | 0.36% | 0.33% |
| pass2 | 10.580 | 10.184 | -3.74% | 0.13% | 0.08% |
| twin | 0.805 | 0.804 | -0.14% | 0.07% | 0.13% |
| glue | 5.493 | 5.498 | +0.08% | 0.13% | 0.12% |
| e2e_fwd | 22.557 | 22.576 | +0.08% | 0.25% | 0.20% |
| e2e_fwdbwd | 86.681 | 86.308 | -0.43% | 0.23% | 0.16% |

### Per-kernel sums, affected subset D128

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 8.024 | 8.055 | +0.39% | 0.34% | 0.66% |
| prod_fwd | 0.970 | 0.971 | +0.18% | 0.18% | 0.15% |
| prod_bwd | 0.538 | 0.539 | +0.07% | 0.09% | 0.19% |
| pass0 | 1.745 | 1.751 | +0.36% | 0.53% | 0.46% |
| pass1 | 20.599 | 20.688 | +0.43% | 0.26% | 0.60% |
| pass2 | 5.040 | 4.642 | -7.89% | 0.07% | 0.08% |
| twin | 0.370 | 0.369 | -0.17% | 0.05% | 0.14% |
| glue | 2.782 | 2.785 | +0.13% | 0.09% | 0.16% |
| e2e_fwd | 10.873 | 10.888 | +0.14% | 0.37% | 0.26% |
| e2e_fwdbwd | 40.383 | 39.976 | -1.01% | 0.51% | 0.17% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 7.067 | 7.018 | -0.70% | 0.56% | 0.30% |
| D128 B1 H24/8 N12288 | causal | stale | 5.990 | 5.881 | -1.82% | 1.03% | 0.25% |
| D128 B1 H24/8 N12288 | full | stale | 11.347 | 11.268 | -0.69% | 0.91% | 1.06% |
| D128 B2 H32/32 N3072 | causal | matched | 1.704 | 1.696 | -0.52% | 0.32% | 0.27% |
| D128 B2 H32/32 N3072 | causal | stale | 1.506 | 1.495 | -0.77% | 0.14% | 0.44% |
| D128 B2 H32/32 N3072 | full | stale | 2.356 | 2.321 | -1.48% | 0.22% | 0.64% |
| D128 B3 H12/4 N6144 | causal | matched | 3.087 | 3.054 | -1.07% | 0.40% | 0.93% |
| D128 B3 H12/4 N6144 | causal | stale | 2.641 | 2.616 | -0.96% | 0.22% | 0.67% |
| D128 B3 H12/4 N6144 | full | stale | 4.684 | 4.627 | -1.21% | 0.18% | 0.51% |
| D256 B1 H12/4 N12288 | causal | matched | 6.903 | 6.904 | +0.02% | 1.33% | 0.52% |
| D256 B1 H12/4 N12288 | causal | stale | 5.939 | 5.966 | +0.45% | 0.79% | 1.35% |
| D256 B1 H12/4 N12288 | full | stale | 11.225 | 11.232 | +0.06% | 0.24% | 0.24% |
| D256 B2 H16/1 N5120 | causal | matched | 3.407 | 3.412 | +0.14% | 0.04% | 0.19% |
| D256 B2 H16/1 N5120 | causal | stale | 3.033 | 3.028 | -0.14% | 0.23% | 0.17% |
| D256 B2 H16/1 N5120 | full | stale | 5.469 | 5.470 | +0.03% | 0.50% | 0.25% |
| D256 B3 H6/2 N6144 | causal | matched | 2.979 | 2.980 | +0.04% | 0.17% | 0.43% |
| D256 B3 H6/2 N6144 | causal | stale | 2.655 | 2.653 | -0.07% | 0.21% | 0.09% |
| D256 B3 H6/2 N6144 | full | stale | 4.688 | 4.686 | -0.04% | 0.22% | 0.18% |

Worst single-shape fwd+bwd change (B vs A): +0.45%
