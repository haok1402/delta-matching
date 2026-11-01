# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/do2 (library sha256 7d491ed1a5560e4e)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.319 | 17.368 | +0.28% | 0.25% | 0.02% |
| prod_fwd | 1.780 | 1.783 | +0.16% | 0.11% | 0.01% |
| prod_bwd | 1.199 | 1.199 | -0.00% | 0.17% | 0.08% |
| pass0 | 3.383 | 3.382 | -0.04% | 0.68% | 0.88% |
| pass1 | 46.756 | 45.861 | -1.91% | 0.16% | 0.06% |
| pass2 | 10.577 | 10.590 | +0.12% | 0.10% | 0.03% |
| twin | 0.805 | 0.805 | +0.04% | 0.06% | 0.17% |
| glue | 5.479 | 5.482 | +0.07% | 0.13% | 0.07% |
| e2e_fwd | 22.578 | 22.619 | +0.18% | 0.12% | 0.30% |
| e2e_fwdbwd | 87.661 | 86.515 | -1.31% | 0.19% | 0.41% |

### Per-kernel sums, affected subset D256

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 9.289 | 9.328 | +0.42% | 0.16% | 0.06% |
| prod_fwd | 0.811 | 0.813 | +0.17% | 0.06% | 0.17% |
| prod_bwd | 0.662 | 0.662 | +0.01% | 0.15% | 0.04% |
| pass0 | 1.635 | 1.642 | +0.40% | 0.28% | 0.46% |
| pass1 | 26.116 | 25.195 | -3.53% | 0.12% | 0.16% |
| pass2 | 5.540 | 5.554 | +0.24% | 0.15% | 0.13% |
| twin | 0.436 | 0.436 | -0.02% | 0.07% | 0.43% |
| glue | 2.705 | 2.706 | +0.06% | 0.12% | 0.09% |
| e2e_fwd | 11.689 | 11.702 | +0.11% | 0.22% | 0.04% |
| e2e_fwdbwd | 47.335 | 46.279 | -2.23% | 0.42% | 0.28% |

### Shapes over the 1.0% fwd+bwd gate

- (128, 3, 12, 4, 6144, 'causal', 'matched'): +1.08% (A reps [3.077, 3.095, 3.085], B reps [3.079, 3.139, 3.119]; spreads 0.59% / 1.93%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 7.029 | 7.018 | -0.17% | 0.61% | 0.55% |
| D128 B1 H24/8 N12288 | causal | stale | 5.946 | 5.965 | +0.32% | 0.19% | 0.29% |
| D128 B1 H24/8 N12288 | full | stale | 11.354 | 11.247 | -0.94% | 1.11% | 1.31% |
| D128 B2 H32/32 N3072 | causal | matched | 1.707 | 1.707 | +0.04% | 0.29% | 0.27% |
| D128 B2 H32/32 N3072 | causal | stale | 1.503 | 1.505 | +0.11% | 0.69% | 0.69% |
| D128 B2 H32/32 N3072 | full | stale | 2.361 | 2.347 | -0.59% | 0.18% | 0.64% |
| D128 B3 H12/4 N6144 | causal | matched | 3.085 | 3.119 | +1.08% | 0.59% | 1.93% |
| D128 B3 H12/4 N6144 | causal | stale | 2.643 | 2.638 | -0.21% | 0.22% | 0.21% |
| D128 B3 H12/4 N6144 | full | stale | 4.698 | 4.692 | -0.13% | 0.63% | 0.08% |
| D256 B1 H12/4 N12288 | causal | matched | 7.031 | 6.837 | -2.77% | 1.49% | 2.49% |
| D256 B1 H12/4 N12288 | causal | stale | 6.121 | 5.966 | -2.53% | 0.30% | 1.16% |
| D256 B1 H12/4 N12288 | full | stale | 11.453 | 11.257 | -1.71% | 0.79% | 1.28% |
| D256 B2 H16/1 N5120 | causal | matched | 3.506 | 3.406 | -2.85% | 0.82% | 0.59% |
| D256 B2 H16/1 N5120 | causal | stale | 3.114 | 3.047 | -2.16% | 0.52% | 0.49% |
| D256 B2 H16/1 N5120 | full | stale | 5.595 | 5.453 | -2.54% | 1.11% | 1.25% |
| D256 B3 H6/2 N6144 | causal | matched | 3.053 | 2.981 | -2.33% | 0.26% | 0.09% |
| D256 B3 H6/2 N6144 | causal | stale | 2.723 | 2.646 | -2.81% | 0.35% | 0.86% |
| D256 B3 H6/2 N6144 | full | stale | 4.739 | 4.685 | -1.13% | 0.75% | 0.58% |

Worst single-shape fwd+bwd change (B vs A): +1.08%
