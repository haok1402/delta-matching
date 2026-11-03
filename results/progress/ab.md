# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r24_s17 (library sha256 02b18ef4fc1b638b)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 74.732 | 64.952 | -13.09% | 0.12% | 0.62% |
| prod_fwd | 1.127 | 1.165 | +3.36% | 0.33% | 0.35% |
| prod_bwd | 0.623 | 0.635 | +2.02% | 0.10% | 0.21% |
| pass0 | 62.606 | 46.828 | -25.20% | 0.34% | 0.64% |
| pass1 | 196.414 | 168.011 | -14.46% | 0.35% | 0.26% |
| pass2 | 41.781 | 38.026 | -8.99% | 0.14% | 0.10% |
| twin | 0.697 | 0.696 | -0.16% | 0.11% | 0.17% |
| glue | 4.998 | 3.189 | -36.19% | 0.09% | 0.13% |
| e2e_fwd | 76.947 | 65.135 | -15.35% | 0.16% | 0.18% |
| e2e_fwdbwd | 382.349 | 322.432 | -15.67% | 0.28% | 0.12% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 122.902 | 103.723 | -15.61% | 0.54% | 0.20% |
| D128 B1 H32/4 N65536 | causal | matched | 246.551 | 208.167 | -15.57% | 0.46% | 0.30% |
| D128 B2 H16/4 N8192 | causal | matched | 4.388 | 3.568 | -18.69% | 0.48% | 0.99% |
| D128 B2 H32/4 N8192 | causal | matched | 8.507 | 6.974 | -18.02% | 1.10% | 4.18% |

Worst single-shape fwd+bwd change (B vs A): -15.57%
