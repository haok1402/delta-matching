# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p1 (library sha256 be70f302b8ed467b)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/nfwd3n2 (library sha256 08225f9613e06a0b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 77.412 | 70.666 | -8.71% | 0.34% | 0.32% |
| prod_fwd | 1.159 | 1.158 | -0.04% | 0.26% | 0.73% |
| prod_bwd | 0.627 | 0.631 | +0.58% | 0.28% | 0.18% |
| pass0 | 58.465 | 58.640 | +0.30% | 0.36% | 0.24% |
| pass1 | 180.248 | 180.821 | +0.32% | 0.41% | 0.15% |
| pass2 | 37.797 | 37.787 | -0.03% | 0.18% | 0.07% |
| twin | 0.697 | 0.697 | -0.02% | 0.05% | 0.13% |
| glue | 3.178 | 3.179 | +0.04% | 0.25% | 0.16% |
| e2e_fwd | 75.733 | 70.042 | -7.52% | 0.43% | 0.43% |
| e2e_fwdbwd | 357.539 | 352.128 | -1.51% | 0.33% | 0.47% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 77.412 -> 70.666 ms (-8.66%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 114.952 | 113.109 | -1.60% | 0.56% | 0.30% |
| D128 B1 H32/4 N65536 | causal | matched | 230.871 | 227.426 | -1.49% | 0.64% | 0.69% |
| D128 B2 H16/4 N8192 | causal | matched | 3.935 | 3.834 | -2.57% | 0.80% | 3.69% |
| D128 B2 H32/4 N8192 | causal | matched | 7.781 | 7.759 | -0.29% | 3.53% | 2.77% |

Worst single-shape fwd+bwd change (B vs A): -0.29%
