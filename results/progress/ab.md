# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_fwd (library sha256 e21133d5e72f7059)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.558 | 67.458 | -10.72% | 0.20% | 0.48% |
| prod_fwd | 1.133 | 1.162 | +2.56% | 0.07% | 0.26% |
| prod_bwd | 0.625 | 0.632 | +1.14% | 0.37% | 0.19% |
| pass0 | 63.178 | 58.763 | -6.99% | 0.10% | 0.32% |
| pass1 | 198.101 | 181.486 | -8.39% | 0.07% | 0.37% |
| pass2 | 41.844 | 37.813 | -9.64% | 0.03% | 0.15% |
| twin | 0.710 | 0.711 | +0.06% | 0.11% | 0.13% |
| glue | 5.006 | 3.174 | -36.59% | 0.10% | 0.10% |
| e2e_fwd | 77.673 | 67.121 | -13.59% | 0.55% | 1.44% |
| e2e_fwdbwd | 385.847 | 350.403 | -9.19% | 0.61% | 0.25% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 123.914 | 112.925 | -8.87% | 1.03% | 0.28% |
| D128 B1 H32/4 N65536 | causal | matched | 248.953 | 225.923 | -9.25% | 0.44% | 0.30% |
| D128 B2 H16/4 N8192 | causal | matched | 4.389 | 3.824 | -12.88% | 0.25% | 1.43% |
| D128 B2 H32/4 N8192 | causal | matched | 8.591 | 7.731 | -10.02% | 0.04% | 2.29% |

Worst single-shape fwd+bwd change (B vs A): -8.87%
