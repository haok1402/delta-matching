# A/B (acceptance run): grid d128m, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r13_prod (library sha256 756158cb1226f4cf)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/h3 (library sha256 756158cb1226f4cf)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 53.255 | 53.404 | +0.28% | 0.59% | 0.51% |
| prod_fwd | 3.952 | 4.035 | +2.09% | 0.21% | 0.25% |
| prod_bwd | 1.521 | 1.526 | +0.31% | 0.25% | 0.11% |
| pass0 | 43.714 | 43.884 | +0.39% | 0.58% | 0.54% |
| pass1 | 128.152 | 128.564 | +0.32% | 0.45% | 0.36% |
| pass2 | 27.567 | 27.571 | +0.02% | 0.02% | 0.04% |
| twin | 1.682 | 1.679 | -0.13% | 0.08% | 0.18% |
| glue | 14.690 | 7.803 | -46.88% | 0.12% | 0.19% |
| e2e_fwd | 62.728 | 56.473 | -9.97% | 0.67% | 0.36% |
| e2e_fwdbwd | 273.769 | 267.862 | -2.16% | 0.85% | 0.30% |

### Acceptance predicates (target glue, D128 tuned)

- target over all D128 shapes: 5.154 -> 1.662 ms (-67.75%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 9.536 -> 6.141 ms (-35.60%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H8/8 N32768 | causal | matched | 15.254 | 15.025 | -1.50% | 0.87% | 0.72% |
| D128 B1 H16/4 N32768 | causal | matched | 30.912 | 30.650 | -0.85% | 0.47% | 0.01% |
| D128 B1 H16/16 N32768 | causal | matched | 31.075 | 30.553 | -1.68% | 0.38% | 0.54% |
| D128 B1 H32/8 N32768 | causal | matched | 61.755 | 61.226 | -0.86% | 0.58% | 0.30% |
| D128 B2 H8/8 N16384 | causal | matched | 8.012 | 7.733 | -3.48% | 1.23% | 0.81% |
| D128 B2 H16/4 N16384 | causal | matched | 15.844 | 15.630 | -1.35% | 0.30% | 0.30% |
| D128 B2 H16/16 N16384 | causal | matched | 16.023 | 15.545 | -2.98% | 0.82% | 0.85% |
| D128 B2 H32/8 N16384 | causal | matched | 33.605 | 33.146 | -1.37% | 1.37% | 2.61% |
| D128 B4 H8/8 N8192 | causal | matched | 4.332 | 4.035 | -6.87% | 0.07% | 2.10% |
| D128 B4 H16/4 N8192 | causal | matched | 8.493 | 8.383 | -1.29% | 2.92% | 0.72% |
| D128 B4 H16/16 N8192 | causal | matched | 8.698 | 8.300 | -4.57% | 1.13% | 0.39% |
| D128 B4 H32/8 N8192 | causal | matched | 17.165 | 16.657 | -2.96% | 0.67% | 0.79% |
| D128 B8 H8/8 N4096 | causal | matched | 2.612 | 2.314 | -11.42% | 0.07% | 0.59% |
| D128 B8 H16/4 N4096 | causal | matched | 4.855 | 4.585 | -5.57% | 0.64% | 1.10% |
| D128 B8 H16/16 N4096 | causal | matched | 5.372 | 4.725 | -12.04% | 18.00% | 4.07% |
| D128 B8 H32/8 N4096 | causal | matched | 9.761 | 9.355 | -4.15% | 0.79% | 0.18% |

Worst single-shape fwd+bwd change (B vs A): -0.85%
