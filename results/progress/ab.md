# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r18_c4 (library sha256 13235f129191b6c2)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.838 | 67.052 | -10.40% | 0.16% | 0.49% |
| prod_fwd | 1.129 | 1.161 | +2.88% | 0.12% | 0.39% |
| prod_bwd | 0.622 | 0.631 | +1.40% | 0.21% | 0.18% |
| pass0 | 62.654 | 50.360 | -19.62% | 0.21% | 0.50% |
| pass1 | 196.558 | 172.707 | -12.13% | 0.27% | 0.42% |
| pass2 | 41.750 | 37.889 | -9.25% | 0.31% | 0.19% |
| twin | 0.711 | 0.711 | -0.00% | 0.12% | 0.12% |
| glue | 4.994 | 3.175 | -36.43% | 0.09% | 0.17% |
| e2e_fwd | 77.535 | 66.702 | -13.97% | 0.34% | 0.21% |
| e2e_fwdbwd | 382.355 | 332.167 | -13.13% | 0.08% | 0.46% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.958 | 106.911 | -13.05% | 0.31% | 0.56% |
| D128 B1 H32/4 N65536 | causal | matched | 246.408 | 214.160 | -13.09% | 0.02% | 0.90% |
| D128 B2 H16/4 N8192 | causal | matched | 4.528 | 3.696 | -18.38% | 5.55% | 1.45% |
| D128 B2 H32/4 N8192 | causal | matched | 8.461 | 7.400 | -12.54% | 1.92% | 1.42% |

Worst single-shape fwd+bwd change (B vs A): -12.54%
