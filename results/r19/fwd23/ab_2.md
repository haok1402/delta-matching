# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r18_c4 (library sha256 13235f129191b6c2)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/S19-fwd-23 (library sha256 9ffc39ca080ed644)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 67.546 | 64.813 | -4.05% | 0.58% | 0.17% |
| prod_fwd | 1.166 | 1.166 | +0.01% | 0.45% | 0.24% |
| prod_bwd | 0.633 | 0.634 | +0.20% | 0.24% | 0.20% |
| pass0 | 50.744 | 51.280 | +1.06% | 0.43% | 0.13% |
| pass1 | 173.816 | 174.380 | +0.32% | 0.35% | 0.20% |
| pass2 | 37.957 | 38.001 | +0.12% | 0.16% | 0.04% |
| twin | 0.696 | 0.697 | +0.03% | 0.10% | 0.17% |
| glue | 3.182 | 3.181 | -0.04% | 0.14% | 0.16% |
| e2e_fwd | 67.168 | 65.280 | -2.81% | 0.28% | 0.40% |
| e2e_fwdbwd | 334.466 | 332.805 | -0.50% | 0.31% | 0.32% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 67.546 -> 64.813 ms (-3.92%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- (128, 2, 16, 4, 8192, 'causal', 'matched'): +1.02% (A reps [3.674, 3.677, 3.69], B reps [3.63, 3.799, 3.715]; spreads 0.45% / 4.55%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 107.173 | 106.670 | -0.47% | 0.18% | 0.33% |
| D128 B1 H32/4 N65536 | causal | matched | 216.257 | 215.094 | -0.54% | 0.56% | 0.34% |
| D128 B2 H16/4 N8192 | causal | matched | 3.677 | 3.715 | +1.02% | 0.45% | 4.55% |
| D128 B2 H32/4 N8192 | causal | matched | 7.358 | 7.326 | -0.43% | 0.21% | 6.20% |

Worst single-shape fwd+bwd change (B vs A): +1.02%
