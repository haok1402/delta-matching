# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_pk (library sha256 5bbf85214faa7022)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p1c4 (library sha256 13235f129191b6c2)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 67.112 | 67.505 | +0.59% | 0.36% | 0.39% |
| prod_fwd | 1.155 | 1.161 | +0.48% | 0.35% | 0.18% |
| prod_bwd | 0.631 | 0.632 | +0.16% | 0.21% | 0.04% |
| pass0 | 52.756 | 50.719 | -3.86% | 0.35% | 0.27% |
| pass1 | 179.627 | 174.099 | -3.08% | 0.26% | 0.36% |
| pass2 | 37.949 | 38.054 | +0.28% | 0.22% | 0.06% |
| twin | 0.695 | 0.694 | -0.07% | 0.11% | 0.11% |
| glue | 3.174 | 3.177 | +0.08% | 0.22% | 0.18% |
| e2e_fwd | 66.820 | 66.856 | +0.05% | 0.44% | 0.07% |
| e2e_fwdbwd | 342.219 | 334.297 | -2.31% | 0.67% | 0.24% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 52.756 -> 50.719 ms (-3.78%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 109.768 | 107.351 | -2.20% | 0.34% | 0.35% |
| D128 B1 H32/4 N65536 | causal | matched | 221.357 | 215.971 | -2.43% | 0.87% | 0.16% |
| D128 B2 H16/4 N8192 | causal | matched | 3.710 | 3.629 | -2.19% | 0.35% | 0.65% |
| D128 B2 H32/4 N8192 | causal | matched | 7.384 | 7.346 | -0.51% | 0.42% | 3.40% |

Worst single-shape fwd+bwd change (B vs A): -0.51%
