# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_c2 (library sha256 e4e37208c530355b)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/c3pass0 (library sha256 228257da8cf5f7f5)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.860 | 75.829 | -0.04% | 0.32% | 0.36% |
| prod_fwd | 1.147 | 1.146 | -0.14% | 0.38% | 0.43% |
| prod_bwd | 0.626 | 0.626 | -0.00% | 0.11% | 0.25% |
| pass0 | 60.229 | 57.452 | -4.61% | 0.35% | 0.54% |
| pass1 | 188.899 | 187.972 | -0.49% | 0.32% | 0.57% |
| pass2 | 37.580 | 37.523 | -0.15% | 0.21% | 0.19% |
| twin | 0.696 | 0.697 | +0.05% | 0.24% | 0.17% |
| glue | 3.164 | 3.168 | +0.12% | 0.19% | 0.35% |
| e2e_fwd | 75.538 | 75.502 | -0.05% | 0.12% | 0.45% |
| e2e_fwdbwd | 367.039 | 363.412 | -0.99% | 0.38% | 0.57% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 60.229 -> 57.452 ms (-7.67%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 117.736 | 116.608 | -0.96% | 0.29% | 0.63% |
| D128 B1 H32/4 N65536 | causal | matched | 237.177 | 234.906 | -0.96% | 0.46% | 0.75% |
| D128 B2 H16/4 N8192 | causal | matched | 4.055 | 3.973 | -2.02% | 0.83% | 1.06% |
| D128 B2 H32/4 N8192 | causal | matched | 8.071 | 7.925 | -1.81% | 1.91% | 0.58% |

Worst single-shape fwd+bwd change (B vs A): -0.96%
