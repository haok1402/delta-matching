# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r14_h3 (library sha256 756158cb1226f4cf)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/c2pass0 (library sha256 e4e37208c530355b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.443 | 75.785 | +0.45% | 0.49% | 0.06% |
| prod_fwd | 1.144 | 1.145 | +0.08% | 0.36% | 0.17% |
| prod_bwd | 0.625 | 0.625 | +0.06% | 0.34% | 0.06% |
| pass0 | 63.060 | 59.686 | -5.35% | 0.28% | 0.32% |
| pass1 | 186.724 | 187.145 | +0.23% | 0.30% | 0.33% |
| pass2 | 37.482 | 37.482 | +0.00% | 0.07% | 0.07% |
| twin | 0.696 | 0.696 | +0.03% | 0.20% | 0.19% |
| glue | 3.157 | 3.160 | +0.08% | 0.28% | 0.06% |
| e2e_fwd | 75.207 | 75.006 | -0.27% | 0.24% | 0.43% |
| e2e_fwdbwd | 367.807 | 365.456 | -0.64% | 0.55% | 0.35% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 63.060 -> 59.686 ms (-5.27%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 118.141 | 117.726 | -0.35% | 0.19% | 0.75% |
| D128 B1 H32/4 N65536 | causal | matched | 237.485 | 235.632 | -0.78% | 0.78% | 0.54% |
| D128 B2 H16/4 N8192 | causal | matched | 4.088 | 4.039 | -1.19% | 0.95% | 1.14% |
| D128 B2 H32/4 N8192 | causal | matched | 8.093 | 8.059 | -0.43% | 1.34% | 1.21% |

Worst single-shape fwd+bwd change (B vs A): -0.35%
