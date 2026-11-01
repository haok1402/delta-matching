# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/do2 (library sha256 7d491ed1a5560e4e)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.139 | 17.142 | +0.02% | 0.04% | 0.26% |
| prod_fwd | 1.776 | 1.774 | -0.08% | 0.11% | 0.22% |
| prod_bwd | 1.197 | 1.197 | -0.00% | 0.21% | 0.12% |
| pass0 | 3.325 | 3.324 | -0.02% | 0.31% | 0.16% |
| pass1 | 46.271 | 45.283 | -2.14% | 0.09% | 0.27% |
| pass2 | 10.526 | 10.523 | -0.02% | 0.02% | 0.03% |
| twin | 0.804 | 0.805 | +0.17% | 0.20% | 0.08% |
| glue | 5.463 | 5.463 | +0.00% | 0.03% | 0.11% |
| e2e_fwd | 22.440 | 22.474 | +0.15% | 0.51% | 0.62% |
| e2e_fwdbwd | 86.706 | 86.029 | -0.78% | 0.24% | 0.25% |

### Per-kernel sums, affected subset D256

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 9.173 | 9.183 | +0.11% | 0.06% | 0.18% |
| prod_fwd | 0.809 | 0.809 | -0.01% | 0.21% | 0.05% |
| prod_bwd | 0.660 | 0.661 | +0.06% | 0.22% | 0.15% |
| pass0 | 1.599 | 1.604 | +0.31% | 0.36% | 0.25% |
| pass1 | 25.805 | 24.834 | -3.76% | 0.06% | 0.22% |
| pass2 | 5.500 | 5.500 | -0.01% | 0.04% | 0.03% |
| twin | 0.435 | 0.435 | +0.09% | 0.37% | 0.13% |
| glue | 2.693 | 2.696 | +0.10% | 0.05% | 0.07% |
| e2e_fwd | 11.606 | 11.648 | +0.36% | 0.39% | 0.22% |
| e2e_fwdbwd | 46.751 | 46.000 | -1.61% | 0.16% | 0.20% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 6.919 | 6.971 | +0.76% | 0.88% | 1.08% |
| D128 B1 H24/8 N12288 | causal | stale | 5.930 | 5.931 | +0.02% | 0.07% | 0.18% |
| D128 B1 H24/8 N12288 | full | stale | 11.168 | 11.167 | -0.02% | 0.71% | 0.66% |
| D128 B2 H32/32 N3072 | causal | matched | 1.700 | 1.692 | -0.45% | 0.59% | 0.71% |
| D128 B2 H32/32 N3072 | causal | stale | 1.493 | 1.495 | +0.11% | 0.88% | 0.89% |
| D128 B2 H32/32 N3072 | full | stale | 2.345 | 2.346 | +0.01% | 0.29% | 0.20% |
| D128 B3 H12/4 N6144 | causal | matched | 3.082 | 3.107 | +0.83% | 0.62% | 1.52% |
| D128 B3 H12/4 N6144 | causal | stale | 2.637 | 2.635 | -0.08% | 0.37% | 0.11% |
| D128 B3 H12/4 N6144 | full | stale | 4.681 | 4.686 | +0.11% | 0.14% | 0.10% |
| D256 B1 H12/4 N12288 | causal | matched | 6.874 | 6.838 | -0.52% | 0.96% | 2.52% |
| D256 B1 H12/4 N12288 | causal | stale | 6.100 | 5.927 | -2.84% | 0.08% | 0.05% |
| D256 B1 H12/4 N12288 | full | stale | 11.270 | 11.113 | -1.39% | 0.09% | 0.88% |
| D256 B2 H16/1 N5120 | causal | matched | 3.471 | 3.392 | -2.29% | 0.16% | 0.09% |
| D256 B2 H16/1 N5120 | causal | stale | 3.086 | 3.023 | -2.03% | 0.08% | 0.70% |
| D256 B2 H16/1 N5120 | full | stale | 5.511 | 5.430 | -1.46% | 0.13% | 0.82% |
| D256 B3 H6/2 N6144 | causal | matched | 3.019 | 2.972 | -1.57% | 0.68% | 0.20% |
| D256 B3 H6/2 N6144 | causal | stale | 2.694 | 2.629 | -2.42% | 0.12% | 0.05% |
| D256 B3 H6/2 N6144 | full | stale | 4.727 | 4.676 | -1.08% | 0.12% | 0.21% |

Worst single-shape fwd+bwd change (B vs A): +0.83%
