# A/B (acceptance run): grid paper, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_pk (library sha256 5bbf85214faa7022)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 75.302 | 67.493 | -10.37% | 1.01% | 0.59% |
| prod_fwd | 1.131 | 1.161 | +2.61% | 0.43% | 0.55% |
| prod_bwd | 0.624 | 0.632 | +1.33% | 0.33% | 0.18% |
| pass0 | 63.185 | 52.880 | -16.31% | 0.64% | 0.48% |
| pass1 | 198.397 | 180.046 | -9.25% | 0.61% | 0.44% |
| pass2 | 41.889 | 37.873 | -9.59% | 0.18% | 0.18% |
| twin | 0.697 | 0.696 | -0.12% | 0.07% | 0.03% |
| glue | 5.010 | 3.181 | -36.51% | 0.16% | 0.09% |
| e2e_fwd | 77.476 | 67.362 | -13.05% | 0.09% | 0.52% |
| e2e_fwdbwd | 385.356 | 342.276 | -11.18% | 0.84% | 0.18% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N65536 | causal | matched | 123.287 | 109.741 | -10.99% | 0.66% | 0.36% |
| D128 B1 H32/4 N65536 | causal | matched | 249.145 | 221.534 | -11.08% | 0.98% | 0.08% |
| D128 B2 H16/4 N8192 | causal | matched | 4.407 | 3.762 | -14.64% | 1.09% | 1.13% |
| D128 B2 H32/4 N8192 | causal | matched | 8.516 | 7.239 | -15.00% | 0.40% | 3.02% |

Worst single-shape fwd+bwd change (B vs A): -10.99%
