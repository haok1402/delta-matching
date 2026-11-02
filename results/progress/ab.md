# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_c2 (library sha256 e4e37208c530355b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.597 | 75.925 | +1.78% | 0.30% | 0.20% |
| prod_fwd | 1.126 | 1.147 | +1.89% | 0.07% | 0.23% |
| prod_bwd | 0.623 | 0.626 | +0.56% | 0.28% | 0.25% |
| pass0 | 62.280 | 59.852 | -3.90% | 0.12% | 0.12% |
| pass1 | 195.496 | 187.648 | -4.01% | 0.09% | 0.14% |
| pass2 | 41.828 | 37.719 | -9.82% | 0.02% | 0.09% |
| twin | 0.696 | 0.695 | -0.02% | 0.08% | 0.03% |
| glue | 4.996 | 3.165 | -36.66% | 0.04% | 0.06% |
| e2e_fwd | 77.054 | 75.587 | -1.90% | 0.69% | 0.43% |
| e2e_fwdbwd | 380.526 | 365.331 | -3.99% | 0.36% | 0.46% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.418 | 117.298 | -4.18% | 0.18% | 0.86% |
| D128 B1 H32/4 N65536 | causal | matched | 245.229 | 235.989 | -3.77% | 0.67% | 0.28% |
| D128 B2 H16/4 N8192 | causal | matched | 4.392 | 4.077 | -7.18% | 0.52% | 0.86% |
| D128 B2 H32/4 N8192 | causal | matched | 8.487 | 7.967 | -6.13% | 0.81% | 2.14% |

Worst single-shape fwd+bwd change (B vs A): -3.77%
