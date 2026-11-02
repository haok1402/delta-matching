# A/B (acceptance run): grid d128m, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r14_h3 (library sha256 756158cb1226f4cf)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/kspf (library sha256 5d05d9a089b64776)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 52.609 | 52.273 | -0.64% | 0.41% | 0.41% |
| prod_fwd | 4.013 | 4.015 | +0.05% | 0.22% | 0.10% |
| prod_bwd | 1.518 | 1.519 | +0.10% | 0.20% | 0.13% |
| pass0 | 43.184 | 43.329 | +0.34% | 0.44% | 0.27% |
| pass1 | 126.382 | 126.787 | +0.32% | 0.56% | 0.36% |
| pass2 | 27.375 | 27.381 | +0.02% | 0.14% | 0.05% |
| twin | 1.679 | 1.678 | -0.06% | 0.07% | 0.03% |
| glue | 7.741 | 7.753 | +0.15% | 0.35% | 0.15% |
| e2e_fwd | 56.115 | 55.732 | -0.68% | 0.49% | 0.29% |
| e2e_fwdbwd | 264.017 | 263.818 | -0.08% | 0.37% | 0.09% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 11.046 -> 11.005 ms (-0.38%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 41.563 -> 41.269 ms (-0.71%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': False, 'tune_shapes': True} -> **REJECT**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H8/8 N32768 | causal | matched | 14.762 | 14.744 | -0.12% | 1.06% | 0.18% |
| D128 B1 H16/4 N32768 | causal | matched | 30.172 | 30.107 | -0.22% | 0.43% | 0.69% |
| D128 B1 H16/16 N32768 | causal | matched | 30.149 | 30.110 | -0.13% | 0.97% | 0.23% |
| D128 B1 H32/8 N32768 | causal | matched | 60.453 | 60.428 | -0.04% | 0.53% | 0.35% |
| D128 B2 H8/8 N16384 | causal | matched | 7.676 | 7.626 | -0.65% | 1.61% | 1.98% |
| D128 B2 H16/4 N16384 | causal | matched | 15.278 | 15.267 | -0.07% | 0.22% | 0.36% |
| D128 B2 H16/16 N16384 | causal | matched | 15.263 | 15.250 | -0.09% | 1.19% | 0.79% |
| D128 B2 H32/8 N16384 | causal | matched | 32.815 | 32.962 | +0.45% | 0.72% | 0.93% |
| D128 B4 H8/8 N8192 | causal | matched | 4.043 | 4.038 | -0.14% | 1.69% | 0.21% |
| D128 B4 H16/4 N8192 | causal | matched | 8.156 | 8.078 | -0.95% | 2.34% | 0.91% |
| D128 B4 H16/16 N8192 | causal | matched | 8.159 | 8.188 | +0.35% | 0.81% | 1.31% |
| D128 B4 H32/8 N8192 | causal | matched | 16.313 | 16.376 | +0.39% | 1.35% | 0.22% |
| D128 B8 H8/8 N4096 | causal | matched | 2.319 | 2.311 | -0.34% | 0.51% | 0.25% |
| D128 B8 H16/4 N4096 | causal | matched | 4.571 | 4.555 | -0.35% | 0.78% | 0.01% |
| D128 B8 H16/16 N4096 | causal | matched | 4.761 | 4.624 | -2.87% | 3.79% | 1.33% |
| D128 B8 H32/8 N4096 | causal | matched | 9.126 | 9.153 | +0.30% | 1.20% | 0.91% |

Worst single-shape fwd+bwd change (B vs A): +0.45%
