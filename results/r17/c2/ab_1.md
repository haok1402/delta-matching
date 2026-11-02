# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r14_h3 (library sha256 756158cb1226f4cf)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/c2pass0 (library sha256 e4e37208c530355b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.717 | 75.953 | +0.31% | 0.18% | 0.26% |
| prod_fwd | 1.147 | 1.150 | +0.28% | 0.15% | 0.05% |
| prod_bwd | 0.626 | 0.626 | +0.05% | 0.11% | 0.10% |
| pass0 | 63.127 | 59.828 | -5.23% | 0.23% | 0.36% |
| pass1 | 186.811 | 187.501 | +0.37% | 0.22% | 0.31% |
| pass2 | 37.774 | 37.822 | +0.13% | 0.06% | 0.05% |
| twin | 0.696 | 0.696 | +0.01% | 0.05% | 0.06% |
| glue | 3.170 | 3.165 | -0.16% | 0.20% | 0.08% |
| e2e_fwd | 75.522 | 75.562 | +0.05% | 1.06% | 0.23% |
| e2e_fwdbwd | 367.828 | 365.490 | -0.64% | 0.02% | 0.21% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 63.127 -> 59.828 ms (-5.21%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 118.227 | 117.502 | -0.61% | 0.09% | 0.20% |
| D128 B1 H32/4 N65536 | causal | matched | 237.417 | 235.883 | -0.65% | 0.09% | 0.22% |
| D128 B2 H16/4 N8192 | causal | matched | 4.089 | 4.100 | +0.26% | 0.22% | 1.34% |
| D128 B2 H32/4 N8192 | causal | matched | 8.096 | 8.005 | -1.12% | 1.09% | 0.74% |

Worst single-shape fwd+bwd change (B vs A): +0.26%
