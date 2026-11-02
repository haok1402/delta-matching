# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p0s (library sha256 11a7eebdcabe2457)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/np1pk (library sha256 5bbf85214faa7022)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 65.851 | 66.232 | +0.58% | 0.33% | 0.06% |
| prod_fwd | 1.147 | 1.149 | +0.18% | 0.13% | 0.12% |
| prod_bwd | 0.628 | 0.629 | +0.19% | 0.10% | 0.15% |
| pass0 | 54.584 | 52.063 | -4.62% | 0.08% | 0.25% |
| pass1 | 180.143 | 177.754 | -1.33% | 0.67% | 0.26% |
| pass2 | 37.650 | 37.673 | +0.06% | 0.22% | 0.08% |
| twin | 0.696 | 0.697 | +0.02% | 0.15% | 0.13% |
| glue | 3.163 | 3.167 | +0.11% | 0.24% | 0.18% |
| e2e_fwd | 66.171 | 66.142 | -0.04% | 0.06% | 0.72% |
| e2e_fwdbwd | 341.998 | 337.585 | -1.29% | 0.69% | 0.57% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 54.584 -> 52.063 ms (-4.50%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 110.018 | 108.271 | -1.59% | 0.46% | 0.70% |
| D128 B1 H32/4 N65536 | causal | matched | 220.677 | 218.201 | -1.12% | 0.73% | 0.94% |
| D128 B2 H16/4 N8192 | causal | matched | 3.809 | 3.734 | -1.96% | 2.04% | 0.44% |
| D128 B2 H32/4 N8192 | causal | matched | 7.494 | 7.378 | -1.54% | 2.62% | 3.52% |

Worst single-shape fwd+bwd change (B vs A): -1.12%
