# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r19_fwd (library sha256 9ffc39ca080ed644)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.351 | 64.223 | -13.62% | 0.09% | 0.26% |
| prod_fwd | 1.126 | 1.160 | +3.05% | 0.23% | 0.06% |
| prod_bwd | 0.622 | 0.633 | +1.75% | 0.51% | 0.10% |
| pass0 | 62.105 | 50.920 | -18.01% | 0.30% | 0.18% |
| pass1 | 194.891 | 172.341 | -11.57% | 0.26% | 0.18% |
| pass2 | 41.682 | 37.804 | -9.30% | 0.12% | 0.08% |
| twin | 0.697 | 0.697 | -0.09% | 0.15% | 0.15% |
| glue | 4.989 | 3.173 | -36.40% | 0.06% | 0.16% |
| e2e_fwd | 76.916 | 64.774 | -15.79% | 0.48% | 0.40% |
| e2e_fwdbwd | 379.880 | 329.338 | -13.30% | 0.20% | 0.13% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.136 | 105.619 | -13.52% | 0.43% | 0.33% |
| D128 B1 H32/4 N65536 | causal | matched | 244.953 | 212.878 | -13.09% | 0.14% | 0.22% |
| D128 B2 H16/4 N8192 | causal | matched | 4.384 | 3.623 | -17.36% | 0.25% | 1.09% |
| D128 B2 H32/4 N8192 | causal | matched | 8.407 | 7.218 | -14.14% | 1.95% | 3.12% |

Worst single-shape fwd+bwd change (B vs A): -13.09%
