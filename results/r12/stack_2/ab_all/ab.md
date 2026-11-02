# A/B (acceptance run): grid all, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r10_psipf (library sha256 e1119aaaa89ca07e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/stack (library sha256 2fb59676beb11a4e)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 203.314 | 204.427 | +0.55% | 0.56% | 0.20% |
| prod_fwd | 16.311 | 16.331 | +0.13% | 0.09% | 0.03% |
| prod_bwd | 8.553 | 8.564 | +0.12% | 0.07% | 0.04% |
| pass0 | 39.100 | 39.325 | +0.58% | 0.60% | 0.05% |
| pass1 | 540.019 | 529.462 | -1.95% | 0.53% | 0.13% |
| pass2 | 107.956 | 108.020 | +0.06% | 0.16% | 0.06% |
| twin | 6.286 | 6.288 | +0.03% | 0.08% | 0.11% |
| glue | 42.017 | 42.097 | +0.19% | 0.13% | 0.03% |
| e2e_fwd | 249.584 | 249.669 | +0.03% | 0.20% | 0.15% |
| e2e_fwdbwd | 960.426 | 951.042 | -0.98% | 0.33% | 0.21% |

### Acceptance predicates (target pass1, D128 tuned)

- target over all D128 shapes: 264.967 -> 254.386 ms (-3.99%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 275.052 -> 275.076 ms (+0.01%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'other_shapes': True, 'other_target': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- (256, 2, 16, 1, 5120, 'causal', 'stale'): +1.20% (A reps [2.99, 2.988, 3.032], B reps [2.988, 3.026, 3.03]; spreads 1.49% / 1.39%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N32768 | causal | matched | 30.894 | 30.324 | -1.84% | 0.99% | 0.18% |
| D128 B1 H16/4 N32768 | causal | stale | 25.752 | 25.226 | -2.04% | 0.43% | 0.34% |
| D128 B1 H16/4 N32768 | full | stale | 50.604 | 49.531 | -2.12% | 0.20% | 0.04% |
| D128 B1 H16/16 N32768 | causal | matched | 31.242 | 30.635 | -1.94% | 1.59% | 0.38% |
| D128 B1 H16/16 N32768 | causal | stale | 25.970 | 25.320 | -2.50% | 0.74% | 0.23% |
| D128 B1 H16/16 N32768 | full | stale | 50.620 | 49.199 | -2.81% | 0.33% | 0.38% |
| D128 B1 H24/8 N12288 | causal | matched | 6.973 | 6.854 | -1.71% | 1.17% | 0.36% |
| D128 B1 H24/8 N12288 | causal | stale | 5.882 | 5.741 | -2.40% | 0.12% | 0.72% |
| D128 B1 H24/8 N12288 | full | stale | 11.085 | 10.882 | -1.83% | 0.27% | 1.21% |
| D128 B2 H16/4 N16384 | causal | matched | 16.019 | 15.862 | -0.98% | 1.03% | 5.14% |
| D128 B2 H16/4 N16384 | causal | stale | 13.228 | 12.960 | -2.02% | 0.51% | 0.56% |
| D128 B2 H16/4 N16384 | full | stale | 25.630 | 25.175 | -1.78% | 0.78% | 0.94% |
| D128 B2 H16/16 N16384 | causal | matched | 16.056 | 15.785 | -1.69% | 1.26% | 0.61% |
| D128 B2 H16/16 N16384 | causal | stale | 13.473 | 13.214 | -1.92% | 2.10% | 0.02% |
| D128 B2 H16/16 N16384 | full | stale | 25.796 | 25.270 | -2.04% | 0.72% | 0.65% |
| D128 B2 H32/32 N3072 | causal | matched | 1.692 | 1.650 | -2.52% | 1.10% | 0.62% |
| D128 B2 H32/32 N3072 | causal | stale | 1.488 | 1.456 | -2.13% | 1.31% | 0.57% |
| D128 B2 H32/32 N3072 | full | stale | 2.335 | 2.260 | -3.23% | 1.23% | 0.58% |
| D128 B3 H12/4 N6144 | causal | matched | 3.038 | 2.985 | -1.75% | 0.84% | 0.40% |
| D128 B3 H12/4 N6144 | causal | stale | 2.622 | 2.561 | -2.32% | 0.69% | 0.56% |
| D128 B3 H12/4 N6144 | full | stale | 4.643 | 4.532 | -2.39% | 0.63% | 1.06% |
| D128 B4 H16/4 N8192 | causal | matched | 8.564 | 8.386 | -2.07% | 1.69% | 0.33% |
| D128 B4 H16/4 N8192 | causal | stale | 7.265 | 7.079 | -2.56% | 0.68% | 0.52% |
| D128 B4 H16/4 N8192 | full | stale | 13.361 | 13.144 | -1.62% | 1.34% | 0.98% |
| D128 B4 H16/16 N8192 | causal | matched | 8.753 | 8.627 | -1.43% | 1.43% | 3.41% |
| D128 B4 H16/16 N8192 | causal | stale | 7.466 | 7.243 | -2.98% | 0.19% | 0.48% |
| D128 B4 H16/16 N8192 | full | stale | 13.564 | 13.288 | -2.03% | 0.79% | 0.49% |
| D128 B8 H16/4 N4096 | causal | matched | 4.925 | 4.855 | -1.43% | 0.04% | 0.74% |
| D128 B8 H16/4 N4096 | causal | stale | 4.347 | 4.244 | -2.36% | 1.39% | 0.19% |
| D128 B8 H16/4 N4096 | full | stale | 7.290 | 7.153 | -1.88% | 0.07% | 1.03% |
| D128 B8 H16/16 N4096 | causal | matched | 5.131 | 5.041 | -1.76% | 0.54% | 0.60% |
| D128 B8 H16/16 N4096 | causal | stale | 4.484 | 4.427 | -1.28% | 0.50% | 0.57% |
| D128 B8 H16/16 N4096 | full | stale | 7.418 | 7.326 | -1.25% | 1.28% | 0.15% |
| D128 B16 H16/4 N2048 | causal | matched | 3.204 | 3.144 | -1.87% | 0.14% | 0.27% |
| D128 B16 H16/4 N2048 | causal | stale | 2.870 | 2.806 | -2.22% | 0.14% | 0.58% |
| D128 B16 H16/4 N2048 | full | stale | 4.306 | 4.212 | -2.17% | 0.09% | 0.36% |
| D128 B16 H16/16 N2048 | causal | matched | 3.434 | 3.383 | -1.50% | 1.39% | 0.26% |
| D128 B16 H16/16 N2048 | causal | stale | 3.042 | 2.989 | -1.74% | 0.41% | 0.24% |
| D128 B16 H16/16 N2048 | full | stale | 4.508 | 4.393 | -2.54% | 0.59% | 0.13% |
| D256 B1 H8/2 N32768 | causal | matched | 29.941 | 29.903 | -0.13% | 0.51% | 1.40% |
| D256 B1 H8/2 N32768 | causal | stale | 25.449 | 25.475 | +0.10% | 0.32% | 0.50% |
| D256 B1 H8/2 N32768 | full | stale | 50.348 | 50.668 | +0.64% | 0.98% | 0.88% |
| D256 B1 H8/8 N32768 | causal | matched | 30.077 | 30.189 | +0.37% | 0.04% | 1.34% |
| D256 B1 H8/8 N32768 | causal | stale | 25.707 | 25.705 | -0.01% | 1.06% | 1.23% |
| D256 B1 H8/8 N32768 | full | stale | 50.677 | 50.706 | +0.06% | 0.80% | 0.64% |
| D256 B1 H12/4 N12288 | causal | matched | 6.822 | 6.799 | -0.34% | 0.50% | 0.71% |
| D256 B1 H12/4 N12288 | causal | stale | 5.941 | 5.908 | -0.56% | 0.30% | 0.54% |
| D256 B1 H12/4 N12288 | full | stale | 11.172 | 11.117 | -0.49% | 0.25% | 0.71% |
| D256 B2 H8/2 N16384 | causal | matched | 15.708 | 15.600 | -0.69% | 0.15% | 0.45% |
| D256 B2 H8/2 N16384 | causal | stale | 13.389 | 13.383 | -0.04% | 1.04% | 0.70% |
| D256 B2 H8/2 N16384 | full | stale | 25.623 | 25.746 | +0.48% | 0.48% | 0.49% |
| D256 B2 H8/8 N16384 | causal | matched | 15.696 | 15.751 | +0.35% | 0.62% | 0.65% |
| D256 B2 H8/8 N16384 | causal | stale | 13.571 | 13.489 | -0.60% | 1.02% | 0.50% |
| D256 B2 H8/8 N16384 | full | stale | 25.781 | 25.879 | +0.38% | 0.68% | 0.85% |
| D256 B2 H16/1 N5120 | causal | matched | 3.383 | 3.392 | +0.28% | 0.31% | 0.61% |
| D256 B2 H16/1 N5120 | causal | stale | 2.990 | 3.026 | +1.20% | 1.49% | 1.39% |
| D256 B2 H16/1 N5120 | full | stale | 5.442 | 5.451 | +0.17% | 0.51% | 0.60% |
| D256 B3 H6/2 N6144 | causal | matched | 2.980 | 2.979 | -0.05% | 0.73% | 0.17% |
| D256 B3 H6/2 N6144 | causal | stale | 2.652 | 2.651 | -0.04% | 0.60% | 0.12% |
| D256 B3 H6/2 N6144 | full | stale | 4.691 | 4.698 | +0.16% | 0.58% | 0.20% |
| D256 B4 H8/2 N8192 | causal | matched | 8.360 | 8.365 | +0.07% | 0.99% | 0.75% |
| D256 B4 H8/2 N8192 | causal | stale | 7.303 | 7.298 | -0.07% | 0.21% | 0.56% |
| D256 B4 H8/2 N8192 | full | stale | 13.484 | 13.458 | -0.19% | 0.11% | 0.60% |
| D256 B4 H8/8 N8192 | causal | matched | 8.594 | 8.550 | -0.51% | 1.93% | 1.49% |
| D256 B4 H8/8 N8192 | causal | stale | 7.479 | 7.480 | +0.01% | 0.23% | 0.52% |
| D256 B4 H8/8 N8192 | full | stale | 13.597 | 13.629 | +0.24% | 0.90% | 0.86% |
| D256 B8 H8/2 N4096 | causal | matched | 4.836 | 4.822 | -0.30% | 0.45% | 1.26% |
| D256 B8 H8/2 N4096 | causal | stale | 4.315 | 4.316 | +0.04% | 0.60% | 0.62% |
| D256 B8 H8/2 N4096 | full | stale | 7.389 | 7.398 | +0.12% | 0.37% | 0.35% |
| D256 B8 H8/8 N4096 | causal | matched | 4.998 | 4.977 | -0.41% | 0.41% | 0.59% |
| D256 B8 H8/8 N4096 | causal | stale | 4.509 | 4.520 | +0.23% | 0.84% | 0.08% |
| D256 B8 H8/8 N4096 | full | stale | 7.545 | 7.525 | -0.28% | 0.52% | 0.78% |
| D256 B16 H8/2 N2048 | causal | matched | 3.089 | 3.084 | -0.17% | 0.07% | 0.59% |
| D256 B16 H8/2 N2048 | causal | stale | 2.826 | 2.819 | -0.26% | 0.30% | 0.34% |
| D256 B16 H8/2 N2048 | full | stale | 4.341 | 4.343 | +0.05% | 0.17% | 0.52% |
| D256 B16 H8/8 N2048 | causal | matched | 3.291 | 3.289 | -0.07% | 1.17% | 0.69% |
| D256 B16 H8/8 N2048 | causal | stale | 2.989 | 3.002 | +0.45% | 0.51% | 0.52% |
| D256 B16 H8/8 N2048 | full | stale | 4.466 | 4.488 | +0.48% | 0.60% | 0.99% |

Worst single-shape fwd+bwd change (B vs A): +1.20%
