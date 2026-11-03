# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r24_stack (library sha256 8b431597e1e38767)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.517 | 64.119 | -13.95% | 0.55% | 0.71% |
| prod_fwd | 1.125 | 1.158 | +2.89% | 0.23% | 0.25% |
| prod_bwd | 0.622 | 0.632 | +1.63% | 0.18% | 0.20% |
| pass0 | 62.265 | 50.818 | -18.38% | 0.30% | 0.44% |
| pass1 | 195.295 | 172.333 | -11.76% | 0.25% | 0.38% |
| pass2 | 41.686 | 37.799 | -9.32% | 0.11% | 0.14% |
| twin | 0.710 | 0.709 | -0.08% | 0.10% | 0.09% |
| glue | 4.992 | 3.173 | -36.43% | 0.10% | 0.07% |
| e2e_fwd | 77.332 | 64.916 | -16.06% | 0.43% | 0.26% |
| e2e_fwdbwd | 381.101 | 329.987 | -13.41% | 0.46% | 0.51% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.276 | 105.969 | -13.34% | 0.35% | 0.67% |
| D128 B1 H32/4 N65536 | causal | matched | 245.915 | 213.141 | -13.33% | 0.58% | 0.45% |
| D128 B2 H16/4 N8192 | causal | matched | 4.392 | 3.646 | -16.98% | 0.86% | 1.06% |
| D128 B2 H32/4 N8192 | causal | matched | 8.518 | 7.231 | -15.11% | 0.95% | 0.19% |

Worst single-shape fwd+bwd change (B vs A): -13.33%
