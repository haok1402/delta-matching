# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r18_c4 (library sha256 13235f129191b6c2)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/S19-fwd-10 (library sha256 401dc8a35a687744)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 66.784 | 64.801 | -2.97% | 0.20% | 0.29% |
| prod_fwd | 1.155 | 1.156 | +0.05% | 0.79% | 0.09% |
| prod_bwd | 0.629 | 0.631 | +0.25% | 0.31% | 0.07% |
| pass0 | 50.262 | 50.515 | +0.50% | 0.31% | 0.39% |
| pass1 | 172.004 | 172.034 | +0.02% | 0.37% | 0.30% |
| pass2 | 37.758 | 37.756 | -0.01% | 0.14% | 0.06% |
| twin | 0.696 | 0.697 | +0.07% | 0.13% | 0.11% |
| glue | 3.172 | 3.172 | -0.00% | 0.16% | 0.09% |
| e2e_fwd | 66.468 | 65.192 | -1.92% | 0.59% | 0.58% |
| e2e_fwdbwd | 331.231 | 329.663 | -0.47% | 0.71% | 0.15% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 66.784 -> 64.801 ms (-2.90%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': False, 'tune_shapes': True, 'complete_set': True} -> **REJECT**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 106.576 | 106.089 | -0.46% | 0.83% | 0.42% |
| D128 B1 H32/4 N65536 | causal | matched | 213.723 | 212.786 | -0.44% | 0.74% | 0.49% |
| D128 B2 H16/4 N8192 | causal | matched | 3.666 | 3.690 | +0.66% | 7.34% | 1.92% |
| D128 B2 H32/4 N8192 | causal | matched | 7.267 | 7.098 | -2.32% | 0.19% | 2.95% |

Worst single-shape fwd+bwd change (B vs A): +0.66%
