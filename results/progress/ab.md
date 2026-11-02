# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r14_h3 (library sha256 756158cb1226f4cf)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.428 | 75.391 | +1.29% | 0.31% | 0.45% |
| prod_fwd | 1.127 | 1.145 | +1.61% | 0.21% | 0.32% |
| prod_bwd | 0.622 | 0.624 | +0.27% | 0.31% | 0.17% |
| pass0 | 62.325 | 63.247 | +1.48% | 0.37% | 0.35% |
| pass1 | 195.392 | 187.113 | -4.24% | 0.39% | 0.33% |
| pass2 | 41.683 | 37.577 | -9.85% | 0.08% | 0.10% |
| twin | 0.711 | 0.711 | -0.00% | 0.01% | 0.09% |
| glue | 4.990 | 3.166 | -36.55% | 0.11% | 0.25% |
| e2e_fwd | 76.690 | 75.208 | -1.93% | 0.83% | 0.43% |
| e2e_fwdbwd | 380.099 | 367.100 | -3.42% | 0.14% | 0.77% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.098 | 118.193 | -3.20% | 0.74% | 0.15% |
| D128 B1 H32/4 N65536 | causal | matched | 245.191 | 236.868 | -3.39% | 0.50% | 1.30% |
| D128 B2 H16/4 N8192 | causal | matched | 4.377 | 4.165 | -4.84% | 0.91% | 3.51% |
| D128 B2 H32/4 N8192 | causal | matched | 8.434 | 7.875 | -6.63% | 0.94% | 2.11% |

Worst single-shape fwd+bwd change (B vs A): -3.20%
