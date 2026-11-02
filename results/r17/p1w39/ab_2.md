# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_c3 (library sha256 228257da8cf5f7f5)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p1w39 (library sha256 be70f302b8ed467b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.414 | 76.703 | +1.71% | 0.34% | 0.18% |
| prod_fwd | 1.142 | 1.155 | +1.16% | 0.07% | 0.25% |
| prod_bwd | 0.624 | 0.626 | +0.37% | 0.19% | 0.06% |
| pass0 | 56.833 | 57.380 | +0.96% | 0.51% | 0.16% |
| pass1 | 186.368 | 177.167 | -4.94% | 0.64% | 0.20% |
| pass2 | 37.437 | 37.511 | +0.20% | 0.15% | 0.09% |
| twin | 0.711 | 0.711 | -0.08% | 0.30% | 0.15% |
| glue | 3.159 | 3.170 | +0.35% | 0.15% | 0.05% |
| e2e_fwd | 74.596 | 74.688 | +0.12% | 0.14% | 0.16% |
| e2e_fwdbwd | 359.575 | 354.005 | -1.55% | 0.47% | 0.28% |

### Acceptance predicates (target pass1, D128 tuned)

- target over all D128 shapes: 186.368 -> 177.167 ms (-4.87%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 115.718 | 113.820 | -1.64% | 0.96% | 0.77% |
| D128 B1 H32/4 N65536 | causal | matched | 232.078 | 228.564 | -1.51% | 0.50% | 0.07% |
| D128 B2 H16/4 N8192 | causal | matched | 3.963 | 3.876 | -2.20% | 0.64% | 1.39% |
| D128 B2 H32/4 N8192 | causal | matched | 7.816 | 7.745 | -0.92% | 1.86% | 1.25% |

Worst single-shape fwd+bwd change (B vs A): -0.92%
