# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_fwd (library sha256 e21133d5e72f7059)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/p0stack (library sha256 11a7eebdcabe2457)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 67.757 | 67.143 | -0.91% | 0.60% | 0.45% |
| prod_fwd | 1.167 | 1.161 | -0.52% | 0.48% | 0.11% |
| prod_bwd | 0.634 | 0.631 | -0.40% | 0.17% | 0.08% |
| pass0 | 58.950 | 55.636 | -5.62% | 0.51% | 0.34% |
| pass1 | 182.263 | 183.361 | +0.60% | 0.48% | 0.45% |
| pass2 | 37.813 | 37.887 | +0.20% | 0.14% | 0.12% |
| twin | 0.697 | 0.697 | +0.04% | 0.06% | 0.16% |
| glue | 3.180 | 3.175 | -0.16% | 0.29% | 0.11% |
| e2e_fwd | 67.379 | 67.342 | -0.05% | 0.58% | 0.18% |
| e2e_fwdbwd | 350.226 | 348.731 | -0.43% | 0.13% | 0.81% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 58.950 -> 55.636 ms (-3.43%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- (128, 2, 32, 4, 8192, 'causal', 'matched'): +3.26% (A reps [7.563, 7.439, 7.269], B reps [7.682, 7.559, 7.691]; spreads 3.95% / 1.71%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 112.459 | 111.846 | -0.55% | 0.27% | 0.33% |
| D128 B1 H32/4 N65536 | causal | matched | 226.484 | 225.346 | -0.50% | 0.19% | 1.15% |
| D128 B2 H16/4 N8192 | causal | matched | 3.843 | 3.858 | +0.39% | 0.38% | 5.30% |
| D128 B2 H32/4 N8192 | causal | matched | 7.439 | 7.682 | +3.26% | 3.95% | 1.71% |

Worst single-shape fwd+bwd change (B vs A): +3.26%
