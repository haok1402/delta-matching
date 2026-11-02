# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p0s (library sha256 11a7eebdcabe2457)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.193 | 66.743 | -10.04% | 0.13% | 0.05% |
| prod_fwd | 1.122 | 1.155 | +2.90% | 0.23% | 0.55% |
| prod_bwd | 0.621 | 0.630 | +1.50% | 0.17% | 0.33% |
| pass0 | 62.105 | 55.216 | -11.09% | 0.11% | 0.20% |
| pass1 | 194.726 | 181.725 | -6.68% | 0.13% | 0.46% |
| pass2 | 41.683 | 37.775 | -9.38% | 0.04% | 0.14% |
| twin | 0.697 | 0.696 | -0.15% | 0.14% | 0.02% |
| glue | 4.992 | 3.172 | -36.47% | 0.01% | 0.15% |
| e2e_fwd | 76.544 | 66.674 | -12.89% | 0.25% | 0.21% |
| e2e_fwdbwd | 380.167 | 346.200 | -8.93% | 0.64% | 0.13% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.159 | 110.993 | -9.14% | 0.61% | 0.51% |
| D128 B1 H32/4 N65536 | causal | matched | 245.101 | 223.871 | -8.66% | 1.10% | 0.13% |
| D128 B2 H16/4 N8192 | causal | matched | 4.378 | 3.798 | -13.23% | 0.50% | 0.27% |
| D128 B2 H32/4 N8192 | causal | matched | 8.530 | 7.538 | -11.64% | 1.01% | 1.61% |

Worst single-shape fwd+bwd change (B vs A): -8.66%
