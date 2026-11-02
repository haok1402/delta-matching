# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p0s (library sha256 11a7eebdcabe2457)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/np1pk (library sha256 5bbf85214faa7022)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 67.249 | 67.423 | +0.26% | 0.30% | 0.61% |
| prod_fwd | 1.153 | 1.159 | +0.51% | 0.29% | 0.51% |
| prod_bwd | 0.629 | 0.631 | +0.22% | 0.17% | 0.21% |
| pass0 | 55.506 | 52.855 | -4.78% | 0.05% | 0.16% |
| pass1 | 182.174 | 179.994 | -1.20% | 0.25% | 0.24% |
| pass2 | 37.886 | 37.935 | +0.13% | 0.03% | 0.09% |
| twin | 0.696 | 0.696 | -0.02% | 0.12% | 0.16% |
| glue | 3.172 | 3.178 | +0.19% | 0.27% | 0.23% |
| e2e_fwd | 67.346 | 67.412 | +0.10% | 0.73% | 0.65% |
| e2e_fwdbwd | 347.508 | 342.103 | -1.56% | 0.34% | 0.26% |

### Acceptance predicates (target pass0, D128 tuned)

- target over all D128 shapes: 55.506 -> 52.855 ms (-4.04%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 111.797 | 110.121 | -1.50% | 0.95% | 0.46% |
| D128 B1 H32/4 N65536 | causal | matched | 224.228 | 220.824 | -1.52% | 0.13% | 0.40% |
| D128 B2 H16/4 N8192 | causal | matched | 3.841 | 3.742 | -2.57% | 3.58% | 0.52% |
| D128 B2 H32/4 N8192 | causal | matched | 7.643 | 7.417 | -2.96% | 4.19% | 3.59% |

Worst single-shape fwd+bwd change (B vs A): -1.50%
