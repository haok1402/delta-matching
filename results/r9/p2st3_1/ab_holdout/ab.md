# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r3_do2 (library sha256 7d491ed1a5560e4e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p2st3 (library sha256 d2eee98774ebac50)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.204 | 17.242 | +0.22% | 0.09% | 0.28% |
| prod_fwd | 1.779 | 1.778 | -0.02% | 0.04% | 0.17% |
| prod_bwd | 1.199 | 1.199 | -0.00% | 0.07% | 0.03% |
| pass0 | 3.358 | 3.362 | +0.11% | 0.65% | 0.59% |
| pass1 | 45.413 | 45.487 | +0.16% | 0.05% | 0.30% |
| pass2 | 10.547 | 10.129 | -3.97% | 0.06% | 0.15% |
| twin | 0.805 | 0.804 | -0.15% | 0.24% | 0.14% |
| glue | 5.472 | 5.476 | +0.07% | 0.06% | 0.10% |
| e2e_fwd | 22.554 | 22.554 | +0.00% | 0.24% | 0.17% |
| e2e_fwdbwd | 86.252 | 85.898 | -0.41% | 0.19% | 0.29% |

### Per-kernel sums, affected subset D128

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 7.990 | 8.010 | +0.25% | 0.23% | 0.15% |
| prod_fwd | 0.967 | 0.967 | +0.02% | 0.10% | 0.20% |
| prod_bwd | 0.537 | 0.537 | +0.03% | 0.09% | 0.13% |
| pass0 | 1.736 | 1.737 | +0.10% | 0.95% | 0.71% |
| pass1 | 20.518 | 20.570 | +0.26% | 0.17% | 0.13% |
| pass2 | 5.032 | 4.609 | -8.41% | 0.13% | 0.01% |
| twin | 0.369 | 0.369 | -0.03% | 0.28% | 0.24% |
| glue | 2.771 | 2.774 | +0.10% | 0.07% | 0.06% |
| e2e_fwd | 10.873 | 10.891 | +0.17% | 0.36% | 0.33% |
| e2e_fwdbwd | 40.196 | 39.820 | -0.94% | 0.48% | 0.35% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 7.082 | 7.003 | -1.12% | 2.27% | 1.96% |
| D128 B1 H24/8 N12288 | causal | stale | 5.963 | 5.873 | -1.50% | 0.71% | 0.66% |
| D128 B1 H24/8 N12288 | full | stale | 11.178 | 11.137 | -0.37% | 0.14% | 0.90% |
| D128 B2 H32/32 N3072 | causal | matched | 1.701 | 1.693 | -0.50% | 0.22% | 0.34% |
| D128 B2 H32/32 N3072 | causal | stale | 1.502 | 1.489 | -0.89% | 0.29% | 0.27% |
| D128 B2 H32/32 N3072 | full | stale | 2.350 | 2.321 | -1.23% | 0.31% | 0.18% |
| D128 B3 H12/4 N6144 | causal | matched | 3.077 | 3.047 | -0.96% | 1.44% | 0.82% |
| D128 B3 H12/4 N6144 | causal | stale | 2.655 | 2.630 | -0.92% | 0.28% | 0.10% |
| D128 B3 H12/4 N6144 | full | stale | 4.688 | 4.627 | -1.31% | 0.51% | 0.55% |
| D256 B1 H12/4 N12288 | causal | matched | 6.808 | 6.848 | +0.59% | 0.84% | 0.63% |
| D256 B1 H12/4 N12288 | causal | stale | 5.926 | 5.930 | +0.05% | 0.10% | 0.18% |
| D256 B1 H12/4 N12288 | full | stale | 11.125 | 11.128 | +0.03% | 0.04% | 0.48% |
| D256 B2 H16/1 N5120 | causal | matched | 3.404 | 3.405 | +0.03% | 0.17% | 0.11% |
| D256 B2 H16/1 N5120 | causal | stale | 3.025 | 3.008 | -0.57% | 0.19% | 0.65% |
| D256 B2 H16/1 N5120 | full | stale | 5.464 | 5.436 | -0.52% | 0.10% | 0.68% |
| D256 B3 H6/2 N6144 | causal | matched | 2.981 | 2.982 | +0.03% | 0.27% | 0.12% |
| D256 B3 H6/2 N6144 | causal | stale | 2.638 | 2.656 | +0.69% | 0.79% | 0.08% |
| D256 B3 H6/2 N6144 | full | stale | 4.685 | 4.686 | +0.02% | 0.13% | 0.08% |

Worst single-shape fwd+bwd change (B vs A): +0.69%
