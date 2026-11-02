# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r21_clean (library sha256 81245944d4ec7858)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.418 | 73.844 | -0.77% | 0.59% | 0.05% |
| prod_fwd | 1.124 | 1.152 | +2.51% | 0.26% | 0.21% |
| prod_bwd | 0.621 | 0.627 | +1.03% | 0.14% | 0.08% |
| pass0 | 62.228 | 54.493 | -12.43% | 0.45% | 0.42% |
| pass1 | 195.136 | 177.271 | -9.16% | 0.45% | 0.33% |
| pass2 | 41.751 | 37.732 | -9.63% | 0.11% | 0.11% |
| twin | 0.697 | 0.696 | -0.14% | 0.04% | 0.07% |
| glue | 4.994 | 3.170 | -36.52% | 0.17% | 0.06% |
| e2e_fwd | 76.629 | 73.031 | -4.70% | 0.30% | 0.17% |
| e2e_fwdbwd | 379.242 | 347.625 | -8.34% | 0.34% | 0.14% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 121.432 | 111.575 | -8.12% | 0.12% | 0.27% |
| D128 B1 H32/4 N65536 | causal | matched | 244.954 | 224.527 | -8.34% | 0.45% | 0.34% |
| D128 B2 H16/4 N8192 | causal | matched | 4.378 | 3.807 | -13.06% | 0.81% | 1.08% |
| D128 B2 H32/4 N8192 | causal | matched | 8.478 | 7.717 | -8.98% | 0.12% | 3.27% |

Worst single-shape fwd+bwd change (B vs A): -8.12%
