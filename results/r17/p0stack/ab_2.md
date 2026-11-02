# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_fwd (library sha256 e21133d5e72f7059)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p0stack (library sha256 11a7eebdcabe2457)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 66.656 | 66.366 | -0.44% | 1.12% | 0.47% |
| prod_fwd | 1.157 | 1.153 | -0.32% | 0.87% | 0.17% |
| prod_bwd | 0.631 | 0.629 | -0.22% | 0.16% | 0.15% |
| pass0 | 57.870 | 54.973 | -5.01% | 0.43% | 0.65% |
| pass1 | 178.522 | 180.563 | +1.14% | 0.62% | 0.03% |
| pass2 | 37.561 | 37.690 | +0.34% | 0.18% | 0.06% |
| twin | 0.696 | 0.697 | +0.04% | 0.16% | 0.04% |
| glue | 3.169 | 3.169 | +0.01% | 0.15% | 0.01% |
| e2e_fwd | 66.756 | 66.829 | +0.11% | 1.58% | 0.36% |
| e2e_fwdbwd | 344.672 | 344.306 | -0.11% | 0.52% | 0.63% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 57.870 -> 54.973 ms (-3.19%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 110.904 | 110.620 | -0.26% | 0.99% | 0.79% |
| D128 B1 H32/4 N65536 | causal | matched | 222.404 | 222.243 | -0.07% | 0.26% | 0.61% |
| D128 B2 H16/4 N8192 | causal | matched | 3.803 | 3.823 | +0.51% | 0.53% | 1.12% |
| D128 B2 H32/4 N8192 | causal | matched | 7.561 | 7.620 | +0.78% | 1.47% | 1.25% |

Worst single-shape fwd+bwd change (B vs A): +0.78%
