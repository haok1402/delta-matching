# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p1 (library sha256 be70f302b8ed467b)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/nfwd3g (library sha256 e21133d5e72f7059)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 77.399 | 67.633 | -12.62% | 0.71% | 0.18% |
| prod_fwd | 1.160 | 1.156 | -0.35% | 0.97% | 0.21% |
| prod_bwd | 0.628 | 0.631 | +0.58% | 0.29% | 0.17% |
| pass0 | 58.693 | 58.735 | +0.07% | 0.89% | 0.17% |
| pass1 | 181.054 | 181.054 | +0.00% | 0.86% | 0.15% |
| pass2 | 37.833 | 37.812 | -0.06% | 0.33% | 0.05% |
| twin | 0.697 | 0.696 | -0.15% | 0.06% | 0.27% |
| glue | 3.179 | 3.181 | +0.06% | 0.14% | 0.06% |
| e2e_fwd | 75.697 | 67.395 | -10.97% | 0.40% | 0.25% |
| e2e_fwdbwd | 358.647 | 349.528 | -2.54% | 0.74% | 0.11% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 77.399 -> 67.633 ms (-12.68%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 114.821 | 112.198 | -2.28% | 0.51% | 1.04% |
| D128 B1 H32/4 N65536 | causal | matched | 232.058 | 225.866 | -2.67% | 0.86% | 0.38% |
| D128 B2 H16/4 N8192 | causal | matched | 3.922 | 3.816 | -2.69% | 1.59% | 1.19% |
| D128 B2 H32/4 N8192 | causal | matched | 7.846 | 7.647 | -2.54% | 1.56% | 0.89% |

Worst single-shape fwd+bwd change (B vs A): -2.28%
