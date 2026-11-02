# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_c3 (library sha256 228257da8cf5f7f5)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p1w39 (library sha256 be70f302b8ed467b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.284 | 76.675 | +1.85% | 0.34% | 0.70% |
| prod_fwd | 1.143 | 1.154 | +1.02% | 0.27% | 0.45% |
| prod_bwd | 0.624 | 0.626 | +0.38% | 0.23% | 0.24% |
| pass0 | 56.810 | 57.545 | +1.29% | 0.42% | 0.33% |
| pass1 | 186.057 | 177.644 | -4.52% | 0.39% | 0.33% |
| pass2 | 37.533 | 37.623 | +0.24% | 0.08% | 0.13% |
| twin | 0.697 | 0.697 | +0.02% | 0.12% | 0.01% |
| glue | 3.160 | 3.172 | +0.35% | 0.31% | 0.34% |
| e2e_fwd | 74.715 | 74.637 | -0.10% | 0.54% | 0.49% |
| e2e_fwdbwd | 358.739 | 352.415 | -1.76% | 0.78% | 0.67% |

### Acceptance predicates (target pass1, D128 tuned)

- target over all D128 shapes: 186.057 -> 177.644 ms (-4.68%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 115.381 | 113.227 | -1.87% | 0.07% | 0.49% |
| D128 B1 H32/4 N65536 | causal | matched | 231.464 | 227.626 | -1.66% | 1.17% | 0.82% |
| D128 B2 H16/4 N8192 | causal | matched | 4.071 | 3.893 | -4.37% | 6.08% | 0.65% |
| D128 B2 H32/4 N8192 | causal | matched | 7.823 | 7.669 | -1.97% | 1.69% | 3.20% |

Worst single-shape fwd+bwd change (B vs A): -1.66%
