# A/B: A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base, B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base, grid one, 3 reps alternating, median per config

| kernel | A ms (sum, median) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 3.033 | 3.023 | -0.32% | 0.94% | 0.05% |
| prod_fwd | 0.259 | 0.259 | +0.05% | 0.08% | 0.26% |
| prod_bwd | 0.193 | 0.193 | +0.11% | 0.09% | 0.06% |
| pass0 | 0.660 | 0.659 | -0.20% | 0.65% | 0.23% |
| pass1 | 7.805 | 7.781 | -0.31% | 1.10% | 0.04% |
| pass2 | 1.891 | 1.891 | -0.01% | 0.06% | 0.03% |
| twin | 0.137 | 0.137 | +0.11% | 0.35% | 0.29% |
| glue | 0.877 | 3.912 | +345.94% | 0.37% | 0.03% |
| e2e_fwd | 3.886 | 3.879 | -0.18% | 0.31% | 1.74% |
| e2e_fwdbwd | 14.913 | 17.936 | +20.27% | 0.33% | 0.75% |

| shape | attn | mode | A fwd+bwd | B fwd+bwd | change | A fwd | B fwd | change |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D128 B2 H16/4 N8192 | causal | matched | 4.392 | 5.387 | +22.66% | 1.097 | 1.089 | -0.73% |
| D128 B2 H16/4 N8192 | causal | stale | 3.728 | 4.747 | +27.32% | 1.055 | 1.057 | +0.28% |
| D128 B2 H16/4 N8192 | full | stale | 6.793 | 7.802 | +14.86% | 1.734 | 1.732 | -0.12% |

Worst single-shape fwd+bwd change (B vs A): +27.32%
