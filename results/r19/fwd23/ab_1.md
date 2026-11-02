# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r18_c4 (library sha256 13235f129191b6c2)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/S19-fwd-23 (library sha256 9ffc39ca080ed644)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 67.935 | 65.501 | -3.58% | 0.91% | 0.47% |
| prod_fwd | 1.167 | 1.172 | +0.41% | 0.34% | 0.36% |
| prod_bwd | 0.633 | 0.636 | +0.49% | 0.13% | 0.29% |
| pass0 | 50.965 | 51.689 | +1.42% | 0.47% | 0.42% |
| pass1 | 174.890 | 175.836 | +0.54% | 0.75% | 0.45% |
| pass2 | 38.048 | 38.134 | +0.22% | 0.32% | 0.21% |
| twin | 0.697 | 0.697 | +0.03% | 0.08% | 0.07% |
| glue | 3.182 | 3.185 | +0.10% | 0.11% | 0.11% |
| e2e_fwd | 67.138 | 65.863 | -1.90% | 0.12% | 0.40% |
| e2e_fwdbwd | 336.496 | 335.440 | -0.31% | 0.34% | 0.20% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 67.935 -> 65.501 ms (-3.51%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 107.885 | 107.328 | -0.52% | 0.88% | 0.35% |
| D128 B1 H32/4 N65536 | causal | matched | 217.562 | 217.094 | -0.21% | 0.28% | 0.40% |
| D128 B2 H16/4 N8192 | causal | matched | 3.674 | 3.638 | -0.97% | 0.81% | 0.31% |
| D128 B2 H32/4 N8192 | causal | matched | 7.375 | 7.380 | +0.07% | 2.02% | 0.59% |

Worst single-shape fwd+bwd change (B vs A): +0.07%
