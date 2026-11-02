# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p1 (library sha256 be70f302b8ed467b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.288 | 76.706 | +3.26% | 0.44% | 0.31% |
| prod_fwd | 1.123 | 1.156 | +2.91% | 0.08% | 0.17% |
| prod_bwd | 0.621 | 0.627 | +0.89% | 0.16% | 0.11% |
| pass0 | 62.266 | 57.566 | -7.55% | 0.08% | 0.16% |
| pass1 | 195.430 | 177.799 | -9.02% | 0.09% | 0.13% |
| pass2 | 41.693 | 37.718 | -9.53% | 0.04% | 0.05% |
| twin | 0.708 | 0.709 | +0.14% | 0.07% | 0.09% |
| glue | 4.980 | 3.172 | -36.31% | 0.08% | 0.02% |
| e2e_fwd | 76.739 | 75.568 | -1.53% | 0.24% | 0.50% |
| e2e_fwdbwd | 380.550 | 353.863 | -7.01% | 0.26% | 0.37% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.217 | 113.668 | -6.99% | 0.59% | 0.53% |
| D128 B1 H32/4 N65536 | causal | matched | 245.504 | 228.513 | -6.92% | 0.18% | 0.54% |
| D128 B2 H16/4 N8192 | causal | matched | 4.370 | 3.891 | -10.97% | 0.94% | 1.18% |
| D128 B2 H32/4 N8192 | causal | matched | 8.460 | 7.791 | -7.91% | 1.07% | 0.58% |

Worst single-shape fwd+bwd change (B vs A): -6.92%
