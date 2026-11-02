# A/B (acceptance run): grid all, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r9_p2st3 (library sha256 d2eee98774ebac50)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/psipf (library sha256 e1119aaaa89ca07e)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 203.173 | 203.381 | +0.10% | 0.51% | 0.20% |
| prod_fwd | 16.325 | 16.341 | +0.10% | 0.31% | 0.04% |
| prod_bwd | 8.563 | 8.570 | +0.08% | 0.16% | 0.06% |
| pass0 | 39.262 | 39.307 | +0.12% | 0.90% | 0.60% |
| pass1 | 539.492 | 539.927 | +0.08% | 0.53% | 0.25% |
| pass2 | 111.767 | 107.990 | -3.38% | 0.41% | 0.08% |
| twin | 6.304 | 6.302 | -0.03% | 0.03% | 0.01% |
| glue | 42.048 | 42.092 | +0.10% | 0.34% | 0.06% |
| e2e_fwd | 248.939 | 249.264 | +0.13% | 0.05% | 0.32% |
| e2e_fwdbwd | 964.717 | 962.490 | -0.23% | 0.37% | 0.05% |

### Acceptance predicates (target pass2, D128 tuned)

- target over all D128 shapes: 56.078 -> 54.316 ms (-3.14%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 55.689 -> 53.674 ms (-3.62%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'other_shapes': True, 'other_target': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N32768 | causal | matched | 31.367 | 31.312 | -0.17% | 1.48% | 0.27% |
| D128 B1 H16/4 N32768 | causal | stale | 25.917 | 25.880 | -0.14% | 0.54% | 0.93% |
| D128 B1 H16/4 N32768 | full | stale | 50.863 | 50.839 | -0.05% | 0.22% | 0.40% |
| D128 B1 H16/16 N32768 | causal | matched | 31.293 | 31.226 | -0.21% | 0.96% | 1.20% |
| D128 B1 H16/16 N32768 | causal | stale | 26.173 | 26.085 | -0.34% | 1.06% | 0.08% |
| D128 B1 H16/16 N32768 | full | stale | 50.841 | 50.891 | +0.10% | 0.78% | 0.25% |
| D128 B1 H24/8 N12288 | causal | matched | 6.912 | 6.953 | +0.60% | 2.29% | 1.06% |
| D128 B1 H24/8 N12288 | causal | stale | 5.875 | 5.857 | -0.31% | 0.06% | 0.13% |
| D128 B1 H24/8 N12288 | full | stale | 11.211 | 11.049 | -1.44% | 0.89% | 1.15% |
| D128 B2 H16/4 N16384 | causal | matched | 16.148 | 16.171 | +0.14% | 3.03% | 3.02% |
| D128 B2 H16/4 N16384 | causal | stale | 13.338 | 13.353 | +0.11% | 1.57% | 0.50% |
| D128 B2 H16/4 N16384 | full | stale | 25.752 | 25.617 | -0.52% | 1.63% | 1.19% |
| D128 B2 H16/16 N16384 | causal | matched | 16.111 | 16.194 | +0.52% | 1.04% | 1.16% |
| D128 B2 H16/16 N16384 | causal | stale | 13.465 | 13.500 | +0.26% | 1.55% | 0.40% |
| D128 B2 H16/16 N16384 | full | stale | 26.058 | 25.962 | -0.37% | 0.99% | 0.30% |
| D128 B2 H32/32 N3072 | causal | matched | 1.694 | 1.688 | -0.34% | 0.25% | 0.34% |
| D128 B2 H32/32 N3072 | causal | stale | 1.487 | 1.484 | -0.21% | 0.79% | 0.23% |
| D128 B2 H32/32 N3072 | full | stale | 2.321 | 2.311 | -0.45% | 0.52% | 0.37% |
| D128 B3 H12/4 N6144 | causal | matched | 3.057 | 3.035 | -0.71% | 0.85% | 0.35% |
| D128 B3 H12/4 N6144 | causal | stale | 2.615 | 2.605 | -0.37% | 0.73% | 0.21% |
| D128 B3 H12/4 N6144 | full | stale | 4.624 | 4.620 | -0.10% | 0.59% | 0.20% |
| D128 B4 H16/4 N8192 | causal | matched | 8.604 | 8.545 | -0.69% | 1.46% | 1.02% |
| D128 B4 H16/4 N8192 | causal | stale | 7.287 | 7.277 | -0.14% | 0.56% | 0.70% |
| D128 B4 H16/4 N8192 | full | stale | 13.398 | 13.412 | +0.10% | 0.22% | 0.81% |
| D128 B4 H16/16 N8192 | causal | matched | 8.803 | 8.702 | -1.15% | 2.45% | 0.81% |
| D128 B4 H16/16 N8192 | causal | stale | 7.422 | 7.474 | +0.70% | 0.89% | 0.07% |
| D128 B4 H16/16 N8192 | full | stale | 13.585 | 13.560 | -0.18% | 0.83% | 1.02% |
| D128 B8 H16/4 N4096 | causal | matched | 4.989 | 4.958 | -0.61% | 0.87% | 0.09% |
| D128 B8 H16/4 N4096 | causal | stale | 4.363 | 4.331 | -0.73% | 0.40% | 0.79% |
| D128 B8 H16/4 N4096 | full | stale | 7.303 | 7.293 | -0.13% | 0.36% | 0.36% |
| D128 B8 H16/16 N4096 | causal | matched | 5.141 | 5.141 | -0.00% | 0.54% | 0.59% |
| D128 B8 H16/16 N4096 | causal | stale | 4.532 | 4.510 | -0.49% | 0.62% | 0.19% |
| D128 B8 H16/16 N4096 | full | stale | 7.499 | 7.440 | -0.80% | 0.40% | 0.31% |
| D128 B16 H16/4 N2048 | causal | matched | 3.218 | 3.207 | -0.33% | 0.45% | 0.67% |
| D128 B16 H16/4 N2048 | causal | stale | 2.882 | 2.864 | -0.61% | 0.27% | 0.35% |
| D128 B16 H16/4 N2048 | full | stale | 4.324 | 4.299 | -0.57% | 0.50% | 0.41% |
| D128 B16 H16/16 N2048 | causal | matched | 3.530 | 3.430 | -2.84% | 3.04% | 4.18% |
| D128 B16 H16/16 N2048 | causal | stale | 3.054 | 3.051 | -0.10% | 0.51% | 0.52% |
| D128 B16 H16/16 N2048 | full | stale | 4.482 | 4.454 | -0.62% | 0.37% | 1.25% |
| D256 B1 H8/2 N32768 | causal | matched | 30.029 | 29.837 | -0.64% | 0.77% | 0.65% |
| D256 B1 H8/2 N32768 | causal | stale | 25.443 | 25.466 | +0.09% | 0.15% | 0.64% |
| D256 B1 H8/2 N32768 | full | stale | 50.385 | 50.508 | +0.24% | 0.23% | 0.73% |
| D256 B1 H8/8 N32768 | causal | matched | 30.410 | 30.419 | +0.03% | 1.55% | 1.29% |
| D256 B1 H8/8 N32768 | causal | stale | 25.648 | 25.655 | +0.03% | 0.65% | 0.58% |
| D256 B1 H8/8 N32768 | full | stale | 50.587 | 50.774 | +0.37% | 0.69% | 0.39% |
| D256 B1 H12/4 N12288 | causal | matched | 6.844 | 6.724 | -1.75% | 0.35% | 0.19% |
| D256 B1 H12/4 N12288 | causal | stale | 5.926 | 5.909 | -0.29% | 0.15% | 0.10% |
| D256 B1 H12/4 N12288 | full | stale | 11.193 | 11.097 | -0.85% | 0.58% | 0.65% |
| D256 B2 H8/2 N16384 | causal | matched | 15.837 | 15.635 | -1.28% | 2.27% | 3.51% |
| D256 B2 H8/2 N16384 | causal | stale | 13.405 | 13.396 | -0.07% | 1.12% | 0.26% |
| D256 B2 H8/2 N16384 | full | stale | 25.832 | 25.716 | -0.45% | 1.50% | 0.73% |
| D256 B2 H8/8 N16384 | causal | matched | 15.967 | 15.918 | -0.31% | 1.69% | 0.38% |
| D256 B2 H8/8 N16384 | causal | stale | 13.549 | 13.520 | -0.22% | 0.49% | 0.43% |
| D256 B2 H8/8 N16384 | full | stale | 26.017 | 25.969 | -0.18% | 0.96% | 0.76% |
| D256 B2 H16/1 N5120 | causal | matched | 3.410 | 3.399 | -0.32% | 0.58% | 0.17% |
| D256 B2 H16/1 N5120 | causal | stale | 3.031 | 3.010 | -0.69% | 0.84% | 0.91% |
| D256 B2 H16/1 N5120 | full | stale | 5.455 | 5.422 | -0.61% | 0.59% | 0.53% |
| D256 B3 H6/2 N6144 | causal | matched | 2.988 | 2.974 | -0.48% | 1.16% | 0.12% |
| D256 B3 H6/2 N6144 | causal | stale | 2.635 | 2.625 | -0.39% | 0.78% | 0.73% |
| D256 B3 H6/2 N6144 | full | stale | 4.695 | 4.669 | -0.56% | 0.67% | 0.17% |
| D256 B4 H8/2 N8192 | causal | matched | 8.502 | 8.436 | -0.78% | 0.89% | 1.50% |
| D256 B4 H8/2 N8192 | causal | stale | 7.332 | 7.270 | -0.84% | 0.75% | 0.18% |
| D256 B4 H8/2 N8192 | full | stale | 13.550 | 13.426 | -0.92% | 0.37% | 0.10% |
| D256 B4 H8/8 N8192 | causal | matched | 8.585 | 8.598 | +0.15% | 1.63% | 0.43% |
| D256 B4 H8/8 N8192 | causal | stale | 7.536 | 7.448 | -1.16% | 0.70% | 0.22% |
| D256 B4 H8/8 N8192 | full | stale | 13.676 | 13.643 | -0.24% | 0.42% | 0.13% |
| D256 B8 H8/2 N4096 | causal | matched | 4.818 | 4.801 | -0.36% | 0.46% | 0.51% |
| D256 B8 H8/2 N4096 | causal | stale | 4.342 | 4.325 | -0.39% | 0.67% | 0.83% |
| D256 B8 H8/2 N4096 | full | stale | 7.379 | 7.356 | -0.31% | 0.36% | 0.07% |
| D256 B8 H8/8 N4096 | causal | matched | 5.011 | 4.990 | -0.42% | 0.46% | 0.24% |
| D256 B8 H8/8 N4096 | causal | stale | 4.511 | 4.490 | -0.46% | 0.14% | 0.10% |
| D256 B8 H8/8 N4096 | full | stale | 7.550 | 7.509 | -0.55% | 0.40% | 0.21% |
| D256 B16 H8/2 N2048 | causal | matched | 3.108 | 3.090 | -0.57% | 0.36% | 0.15% |
| D256 B16 H8/2 N2048 | causal | stale | 2.840 | 2.821 | -0.67% | 0.46% | 0.10% |
| D256 B16 H8/2 N2048 | full | stale | 4.356 | 4.325 | -0.71% | 0.63% | 0.07% |
| D256 B16 H8/8 N2048 | causal | matched | 3.300 | 3.293 | -0.23% | 0.24% | 0.82% |
| D256 B16 H8/8 N2048 | causal | stale | 3.013 | 2.996 | -0.57% | 0.42% | 0.23% |
| D256 B16 H8/8 N2048 | full | stale | 4.479 | 4.451 | -0.64% | 0.26% | 0.24% |

Worst single-shape fwd+bwd change (B vs A): +0.70%
