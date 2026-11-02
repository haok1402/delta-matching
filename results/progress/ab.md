# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_c3 (library sha256 228257da8cf5f7f5)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.099 | 75.849 | +1.00% | 0.56% | 0.35% |
| prod_fwd | 1.133 | 1.148 | +1.35% | 0.49% | 0.44% |
| prod_bwd | 0.624 | 0.626 | +0.32% | 0.15% | 0.05% |
| pass0 | 62.883 | 57.190 | -9.05% | 0.71% | 0.38% |
| pass1 | 197.235 | 187.586 | -4.89% | 0.73% | 0.35% |
| pass2 | 41.887 | 37.582 | -10.28% | 0.24% | 0.14% |
| twin | 0.697 | 0.697 | -0.07% | 0.04% | 0.10% |
| glue | 5.006 | 3.166 | -36.76% | 0.11% | 0.26% |
| e2e_fwd | 77.180 | 75.351 | -2.37% | 0.19% | 0.71% |
| e2e_fwdbwd | 383.294 | 363.733 | -5.10% | 0.44% | 0.45% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.732 | 116.749 | -4.87% | 0.92% | 0.24% |
| D128 B1 H32/4 N65536 | causal | matched | 247.632 | 235.130 | -5.05% | 0.51% | 0.75% |
| D128 B2 H16/4 N8192 | causal | matched | 4.395 | 4.001 | -8.98% | 0.70% | 0.69% |
| D128 B2 H32/4 N8192 | causal | matched | 8.534 | 7.853 | -7.99% | 0.97% | 3.76% |

Worst single-shape fwd+bwd change (B vs A): -4.87%
