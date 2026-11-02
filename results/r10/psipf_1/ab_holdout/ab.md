# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r9_p2st3 (library sha256 d2eee98774ebac50)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/psipf (library sha256 e1119aaaa89ca07e)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.242 | 17.254 | +0.07% | 0.18% | 0.09% |
| prod_fwd | 1.779 | 1.779 | +0.02% | 0.14% | 0.12% |
| prod_bwd | 1.199 | 1.200 | +0.10% | 0.07% | 0.03% |
| pass0 | 3.386 | 3.379 | -0.22% | 0.17% | 0.53% |
| pass1 | 45.510 | 45.493 | -0.04% | 0.23% | 0.14% |
| pass2 | 10.133 | 9.820 | -3.09% | 0.07% | 0.06% |
| twin | 0.805 | 0.804 | -0.05% | 0.04% | 0.03% |
| glue | 5.481 | 5.489 | +0.13% | 0.02% | 0.05% |
| e2e_fwd | 22.578 | 22.588 | +0.04% | 0.12% | 0.36% |
| e2e_fwdbwd | 85.963 | 85.710 | -0.30% | 0.21% | 0.34% |

### Per-kernel sums, affected subset D256

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 9.224 | 9.237 | +0.15% | 0.22% | 0.19% |
| prod_fwd | 0.811 | 0.811 | +0.03% | 0.09% | 0.04% |
| prod_bwd | 0.662 | 0.662 | +0.06% | 0.11% | 0.01% |
| pass0 | 1.642 | 1.643 | +0.09% | 0.10% | 0.14% |
| pass1 | 24.910 | 24.922 | +0.05% | 0.16% | 0.15% |
| pass2 | 5.519 | 5.330 | -3.41% | 0.12% | 0.07% |
| twin | 0.435 | 0.435 | +0.06% | 0.14% | 0.27% |
| glue | 2.704 | 2.708 | +0.18% | 0.07% | 0.08% |
| e2e_fwd | 11.678 | 11.671 | -0.07% | 0.02% | 0.33% |
| e2e_fwdbwd | 46.122 | 45.968 | -0.33% | 0.21% | 0.40% |

### Shapes over the 1.0% fwd+bwd gate

- (128, 3, 12, 4, 6144, 'causal', 'matched'): +2.40% (A reps [3.097, 3.048, 3.043], B reps [3.035, 3.121, 3.173]; spreads 1.74% / 4.43%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 6.998 | 6.973 | -0.35% | 1.62% | 0.56% |
| D128 B1 H24/8 N12288 | causal | stale | 5.868 | 5.856 | -0.21% | 0.02% | 1.06% |
| D128 B1 H24/8 N12288 | full | stale | 11.136 | 11.038 | -0.88% | 0.88% | 0.37% |
| D128 B2 H32/32 N3072 | causal | matched | 1.702 | 1.686 | -0.97% | 0.76% | 0.43% |
| D128 B2 H32/32 N3072 | causal | stale | 1.491 | 1.486 | -0.30% | 0.47% | 0.74% |
| D128 B2 H32/32 N3072 | full | stale | 2.335 | 2.324 | -0.47% | 0.99% | 0.46% |
| D128 B3 H12/4 N6144 | causal | matched | 3.048 | 3.121 | +2.40% | 1.74% | 4.43% |
| D128 B3 H12/4 N6144 | causal | stale | 2.621 | 2.612 | -0.33% | 0.97% | 0.57% |
| D128 B3 H12/4 N6144 | full | stale | 4.643 | 4.645 | +0.04% | 0.72% | 0.67% |
| D256 B1 H12/4 N12288 | causal | matched | 6.828 | 6.799 | -0.42% | 0.59% | 0.38% |
| D256 B1 H12/4 N12288 | causal | stale | 5.927 | 5.915 | -0.21% | 0.09% | 0.80% |
| D256 B1 H12/4 N12288 | full | stale | 11.129 | 11.097 | -0.29% | 0.70% | 0.19% |
| D256 B2 H16/1 N5120 | causal | matched | 3.420 | 3.411 | -0.28% | 0.44% | 0.22% |
| D256 B2 H16/1 N5120 | causal | stale | 3.012 | 3.012 | -0.02% | 0.66% | 1.22% |
| D256 B2 H16/1 N5120 | full | stale | 5.438 | 5.446 | +0.14% | 0.86% | 0.69% |
| D256 B3 H6/2 N6144 | causal | matched | 3.004 | 2.990 | -0.47% | 0.21% | 0.89% |
| D256 B3 H6/2 N6144 | causal | stale | 2.649 | 2.636 | -0.49% | 0.59% | 0.54% |
| D256 B3 H6/2 N6144 | full | stale | 4.715 | 4.663 | -1.09% | 0.76% | 0.91% |

Worst single-shape fwd+bwd change (B vs A): +2.40%
