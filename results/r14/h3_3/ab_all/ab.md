# A/B (acceptance run): grid d128m, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r13_prod (library sha256 756158cb1226f4cf)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/h3 (library sha256 756158cb1226f4cf)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 53.259 | 53.354 | +0.18% | 1.50% | 0.34% |
| prod_fwd | 3.950 | 4.031 | +2.04% | 0.55% | 0.25% |
| prod_bwd | 1.519 | 1.525 | +0.38% | 0.47% | 0.32% |
| pass0 | 43.678 | 43.745 | +0.15% | 1.48% | 0.44% |
| pass1 | 127.925 | 128.099 | +0.14% | 1.60% | 0.45% |
| pass2 | 27.498 | 27.488 | -0.04% | 0.44% | 0.10% |
| twin | 1.667 | 1.666 | -0.07% | 0.22% | 0.10% |
| glue | 14.693 | 7.769 | -47.12% | 0.71% | 0.20% |
| e2e_fwd | 62.810 | 56.555 | -9.96% | 0.13% | 0.38% |
| e2e_fwdbwd | 272.809 | 267.253 | -2.04% | 1.03% | 0.22% |

### Acceptance predicates (target glue, D128 tuned)

- target over all D128 shapes: 5.138 -> 1.657 ms (-67.75%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 9.554 -> 6.112 ms (-36.03%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H8/8 N32768 | causal | matched | 15.260 | 14.893 | -2.40% | 1.23% | 0.76% |
| D128 B1 H16/4 N32768 | causal | matched | 30.625 | 30.441 | -0.60% | 0.84% | 0.06% |
| D128 B1 H16/16 N32768 | causal | matched | 30.982 | 30.555 | -1.38% | 1.38% | 0.75% |
| D128 B1 H32/8 N32768 | causal | matched | 61.497 | 61.115 | -0.62% | 0.81% | 0.42% |
| D128 B2 H8/8 N16384 | causal | matched | 7.956 | 7.747 | -2.62% | 0.87% | 1.30% |
| D128 B2 H16/4 N16384 | causal | matched | 15.796 | 15.506 | -1.84% | 1.47% | 0.16% |
| D128 B2 H16/16 N16384 | causal | matched | 16.002 | 15.444 | -3.49% | 1.32% | 0.11% |
| D128 B2 H32/8 N16384 | causal | matched | 33.759 | 33.549 | -0.62% | 0.45% | 1.49% |
| D128 B4 H8/8 N8192 | causal | matched | 4.332 | 4.049 | -6.54% | 1.09% | 0.14% |
| D128 B4 H16/4 N8192 | causal | matched | 8.442 | 8.256 | -2.21% | 1.68% | 1.38% |
| D128 B4 H16/16 N8192 | causal | matched | 8.703 | 8.226 | -5.49% | 3.21% | 1.21% |
| D128 B4 H32/8 N8192 | causal | matched | 17.048 | 16.574 | -2.78% | 1.47% | 0.71% |
| D128 B8 H8/8 N4096 | causal | matched | 2.634 | 2.330 | -11.53% | 0.21% | 0.67% |
| D128 B8 H16/4 N4096 | causal | matched | 4.858 | 4.586 | -5.61% | 0.08% | 0.20% |
| D128 B8 H16/16 N4096 | causal | matched | 5.181 | 4.720 | -8.91% | 4.23% | 2.28% |
| D128 B8 H32/8 N4096 | causal | matched | 9.735 | 9.263 | -4.85% | 0.87% | 0.74% |

Worst single-shape fwd+bwd change (B vs A): -0.60%
