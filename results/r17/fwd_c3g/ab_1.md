# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p1 (library sha256 be70f302b8ed467b)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/nfwd3g (library sha256 e21133d5e72f7059)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 77.333 | 67.140 | -13.18% | 0.08% | 0.37% |
| prod_fwd | 1.161 | 1.159 | -0.17% | 0.04% | 0.09% |
| prod_bwd | 0.629 | 0.632 | +0.58% | 0.15% | 0.09% |
| pass0 | 57.960 | 58.337 | +0.65% | 0.20% | 0.18% |
| pass1 | 178.879 | 179.889 | +0.56% | 0.22% | 0.18% |
| pass2 | 37.689 | 37.729 | +0.11% | 0.09% | 0.05% |
| twin | 0.697 | 0.696 | -0.19% | 0.01% | 0.05% |
| glue | 3.177 | 3.173 | -0.12% | 0.07% | 0.29% |
| e2e_fwd | 75.795 | 67.141 | -11.42% | 0.18% | 0.54% |
| e2e_fwdbwd | 356.776 | 347.357 | -2.64% | 0.55% | 0.23% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 77.333 -> 67.140 ms (-12.89%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 114.552 | 111.181 | -2.94% | 0.52% | 0.49% |
| D128 B1 H32/4 N65536 | causal | matched | 230.527 | 224.851 | -2.46% | 0.53% | 0.19% |
| D128 B2 H16/4 N8192 | causal | matched | 3.940 | 3.798 | -3.58% | 3.58% | 0.73% |
| D128 B2 H32/4 N8192 | causal | matched | 7.757 | 7.526 | -2.98% | 0.51% | 2.39% |

Worst single-shape fwd+bwd change (B vs A): -2.46%
