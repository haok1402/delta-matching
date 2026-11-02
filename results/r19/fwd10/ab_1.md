# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r18_c4 (library sha256 13235f129191b6c2)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/S19-fwd-10 (library sha256 401dc8a35a687744)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 66.641 | 64.646 | -2.99% | 0.56% | 0.48% |
| prod_fwd | 1.156 | 1.158 | +0.12% | 0.43% | 0.42% |
| prod_bwd | 0.630 | 0.631 | +0.15% | 0.26% | 0.18% |
| pass0 | 50.042 | 50.515 | +0.94% | 0.42% | 0.53% |
| pass1 | 171.809 | 172.407 | +0.35% | 0.24% | 0.62% |
| pass2 | 37.733 | 37.767 | +0.09% | 0.07% | 0.20% |
| twin | 0.696 | 0.697 | +0.10% | 0.24% | 0.09% |
| glue | 3.172 | 3.173 | +0.04% | 0.10% | 0.19% |
| e2e_fwd | 66.346 | 64.664 | -2.54% | 0.93% | 0.38% |
| e2e_fwdbwd | 330.168 | 329.035 | -0.34% | 0.73% | 0.23% |

### Acceptance predicates (target fwd, D128 tuned)

- target over all D128 shapes: 66.641 -> 64.646 ms (-3.24%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 0.000 -> 0.000 ms (+0.00%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'complete_set': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 106.368 | 105.225 | -1.08% | 1.25% | 0.48% |
| D128 B1 H32/4 N65536 | causal | matched | 212.849 | 212.877 | +0.01% | 0.39% | 0.20% |
| D128 B2 H16/4 N8192 | causal | matched | 3.661 | 3.671 | +0.27% | 0.18% | 0.46% |
| D128 B2 H32/4 N8192 | causal | matched | 7.290 | 7.263 | -0.37% | 3.47% | 1.96% |

Worst single-shape fwd+bwd change (B vs A): +0.27%
