# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_pk (library sha256 5bbf85214faa7022)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p1c4 (library sha256 13235f129191b6c2)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 67.128 | 67.432 | +0.45% | 0.56% | 0.25% |
| prod_fwd | 1.156 | 1.160 | +0.32% | 0.30% | 0.10% |
| prod_bwd | 0.631 | 0.632 | +0.18% | 0.08% | 0.05% |
| pass0 | 52.631 | 50.760 | -3.55% | 0.25% | 0.01% |
| pass1 | 179.482 | 173.545 | -3.31% | 0.23% | 0.31% |
| pass2 | 37.936 | 37.996 | +0.16% | 0.07% | 0.09% |
| twin | 0.694 | 0.695 | +0.01% | 0.08% | 0.17% |
| glue | 3.173 | 3.175 | +0.07% | 0.07% | 0.03% |
| e2e_fwd | 66.847 | 66.977 | +0.20% | 0.38% | 0.22% |
| e2e_fwdbwd | 341.628 | 334.089 | -2.21% | 0.17% | 0.32% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 52.631 -> 50.760 ms (-3.53%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 109.822 | 107.444 | -2.17% | 0.69% | 0.32% |
| D128 B1 H32/4 N65536 | causal | matched | 220.644 | 215.745 | -2.22% | 0.43% | 0.34% |
| D128 B2 H16/4 N8192 | causal | matched | 3.735 | 3.638 | -2.60% | 0.86% | 0.47% |
| D128 B2 H32/4 N8192 | causal | matched | 7.427 | 7.261 | -2.23% | 3.91% | 1.30% |

Worst single-shape fwd+bwd change (B vs A): -2.17%
