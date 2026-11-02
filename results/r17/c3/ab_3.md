# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_c2 (library sha256 e4e37208c530355b)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/c3pass0 (library sha256 228257da8cf5f7f5)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 76.291 | 75.991 | -0.39% | 0.73% | 0.56% |
| prod_fwd | 1.151 | 1.148 | -0.27% | 0.60% | 0.25% |
| prod_bwd | 0.626 | 0.626 | -0.02% | 0.25% | 0.30% |
| pass0 | 60.063 | 57.176 | -4.81% | 0.67% | 0.32% |
| pass1 | 188.448 | 187.553 | -0.48% | 0.67% | 0.35% |
| pass2 | 37.667 | 37.602 | -0.17% | 0.32% | 0.18% |
| twin | 0.711 | 0.711 | +0.04% | 0.04% | 0.28% |
| glue | 3.172 | 3.165 | -0.25% | 0.13% | 0.20% |
| e2e_fwd | 76.332 | 75.975 | -0.47% | 1.04% | 0.77% |
| e2e_fwdbwd | 366.354 | 362.742 | -0.99% | 0.49% | 0.56% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 60.063 -> 57.176 ms (-7.64%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 118.104 | 116.898 | -1.02% | 0.98% | 0.55% |
| D128 B1 H32/4 N65536 | causal | matched | 236.148 | 234.045 | -0.89% | 0.35% | 0.81% |
| D128 B2 H16/4 N8192 | causal | matched | 4.041 | 3.974 | -1.65% | 0.36% | 0.50% |
| D128 B2 H32/4 N8192 | causal | matched | 8.060 | 7.826 | -2.91% | 2.32% | 3.18% |

Worst single-shape fwd+bwd change (B vs A): -0.89%
