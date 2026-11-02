# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r14_h3 (library sha256 756158cb1226f4cf)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/kspf (library sha256 5d05d9a089b64776)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 76.313 | 75.957 | -0.47% | 0.48% | 0.18% |
| prod_fwd | 1.150 | 1.150 | -0.06% | 0.07% | 0.26% |
| prod_bwd | 0.627 | 0.627 | -0.03% | 0.29% | 0.09% |
| pass0 | 63.856 | 63.618 | -0.37% | 0.15% | 0.18% |
| pass1 | 188.918 | 188.294 | -0.33% | 0.12% | 0.17% |
| pass2 | 37.742 | 37.677 | -0.17% | 0.01% | 0.05% |
| twin | 0.697 | 0.696 | -0.05% | 0.10% | 0.24% |
| glue | 3.167 | 3.165 | -0.04% | 0.24% | 0.27% |
| e2e_fwd | 76.401 | 76.096 | -0.40% | 0.17% | 0.36% |
| e2e_fwdbwd | 371.867 | 370.418 | -0.39% | 0.28% | 0.25% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 76.313 -> 75.957 ms (-0.51%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': False, 'tune_shapes': True, 'complete_set': True} -> **REJECT**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 119.196 | 119.216 | +0.02% | 0.76% | 0.35% |
| D128 B1 H32/4 N65536 | causal | matched | 240.408 | 238.947 | -0.61% | 0.63% | 0.19% |
| D128 B2 H16/4 N8192 | causal | matched | 4.097 | 4.089 | -0.19% | 0.64% | 0.90% |
| D128 B2 H32/4 N8192 | causal | matched | 8.165 | 8.166 | +0.00% | 0.44% | 1.74% |

Worst single-shape fwd+bwd change (B vs A): +0.02%
