# A/B (acceptance run): grid all, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r10_psipf (library sha256 e1119aaaa89ca07e)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/wt/stack (library sha256 2fb59676beb11a4e)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 204.933 | 205.899 | +0.47% | 0.58% | 0.51% |
| prod_fwd | 16.366 | 16.396 | +0.19% | 0.21% | 0.15% |
| prod_bwd | 8.577 | 8.587 | +0.12% | 0.09% | 0.13% |
| pass0 | 39.714 | 39.872 | +0.40% | 0.75% | 0.76% |
| pass1 | 544.696 | 533.199 | -2.11% | 0.65% | 0.54% |
| pass2 | 108.891 | 108.850 | -0.04% | 0.35% | 0.33% |
| twin | 6.336 | 6.334 | -0.04% | 0.05% | 0.01% |
| glue | 42.120 | 42.204 | +0.20% | 0.23% | 0.23% |
| e2e_fwd | 249.804 | 250.636 | +0.33% | 0.19% | 0.20% |
| e2e_fwdbwd | 969.264 | 958.405 | -1.12% | 0.26% | 0.30% |

### Acceptance predicates (target pass1, D128 tuned)

- target over all D128 shapes: 267.181 -> 256.414 ms (-4.03%; needs <= -3.0%)
- D128 shapes slower beyond spread: none
- other head dims: target 277.515 -> 276.785 ms (-0.26%, slower beyond spread: False); shapes slower beyond spread: none
- checks {'target_gain': True, 'tune_shapes': True, 'other_shapes': True, 'other_target': True} -> **ACCEPT (this job)**

### Shapes over the 1.0% fwd+bwd gate

- (256, 4, 8, 8, 8192, 'causal', 'matched'): +1.11% (A reps [8.622, 8.476, 8.714], B reps [8.648, 8.783, 8.717]; spreads 2.77% / 1.54%) (within spread)

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H16/4 N32768 | causal | matched | 31.341 | 30.802 | -1.72% | 0.71% | 0.36% |
| D128 B1 H16/4 N32768 | causal | stale | 26.076 | 25.520 | -2.13% | 0.87% | 0.52% |
| D128 B1 H16/4 N32768 | full | stale | 51.275 | 50.086 | -2.32% | 0.65% | 0.51% |
| D128 B1 H16/16 N32768 | causal | matched | 31.659 | 30.880 | -2.46% | 1.37% | 0.14% |
| D128 B1 H16/16 N32768 | causal | stale | 26.214 | 25.676 | -2.05% | 0.92% | 0.25% |
| D128 B1 H16/16 N32768 | full | stale | 51.099 | 49.811 | -2.52% | 0.56% | 0.46% |
| D128 B1 H24/8 N12288 | causal | matched | 6.983 | 6.918 | -0.93% | 1.50% | 0.80% |
| D128 B1 H24/8 N12288 | causal | stale | 5.890 | 5.749 | -2.39% | 0.58% | 1.82% |
| D128 B1 H24/8 N12288 | full | stale | 11.166 | 10.950 | -1.93% | 0.32% | 0.23% |
| D128 B2 H16/4 N16384 | causal | matched | 16.343 | 16.156 | -1.14% | 1.72% | 2.40% |
| D128 B2 H16/4 N16384 | causal | stale | 13.439 | 13.208 | -1.72% | 0.16% | 1.74% |
| D128 B2 H16/4 N16384 | full | stale | 25.813 | 25.419 | -1.53% | 0.68% | 0.41% |
| D128 B2 H16/16 N16384 | causal | matched | 16.297 | 15.919 | -2.32% | 0.64% | 0.56% |
| D128 B2 H16/16 N16384 | causal | stale | 13.640 | 13.350 | -2.12% | 0.73% | 0.60% |
| D128 B2 H16/16 N16384 | full | stale | 26.146 | 25.604 | -2.07% | 0.99% | 0.74% |
| D128 B2 H32/32 N3072 | causal | matched | 1.684 | 1.655 | -1.72% | 13.24% | 0.44% |
| D128 B2 H32/32 N3072 | causal | stale | 1.483 | 1.456 | -1.81% | 6.69% | 39.42% |
| D128 B2 H32/32 N3072 | full | stale | 2.309 | 2.257 | -2.25% | 0.60% | 0.71% |
| D128 B3 H12/4 N6144 | causal | matched | 3.039 | 2.968 | -2.33% | 0.39% | 0.38% |
| D128 B3 H12/4 N6144 | causal | stale | 2.612 | 2.545 | -2.57% | 0.27% | 0.44% |
| D128 B3 H12/4 N6144 | full | stale | 4.612 | 4.501 | -2.41% | 0.34% | 0.55% |
| D128 B4 H16/4 N8192 | causal | matched | 8.665 | 8.514 | -1.75% | 0.51% | 0.92% |
| D128 B4 H16/4 N8192 | causal | stale | 7.284 | 7.104 | -2.46% | 0.35% | 1.98% |
| D128 B4 H16/4 N8192 | full | stale | 13.443 | 13.183 | -1.93% | 0.77% | 1.43% |
| D128 B4 H16/16 N8192 | causal | matched | 8.846 | 8.757 | -1.01% | 1.50% | 0.49% |
| D128 B4 H16/16 N8192 | causal | stale | 7.468 | 7.263 | -2.75% | 1.24% | 0.58% |
| D128 B4 H16/16 N8192 | full | stale | 13.650 | 13.406 | -1.79% | 0.63% | 1.11% |
| D128 B8 H16/4 N4096 | causal | matched | 4.962 | 4.887 | -1.51% | 0.64% | 0.82% |
| D128 B8 H16/4 N4096 | causal | stale | 4.326 | 4.228 | -2.27% | 0.60% | 0.08% |
| D128 B8 H16/4 N4096 | full | stale | 7.341 | 7.157 | -2.51% | 0.66% | 0.43% |
| D128 B8 H16/16 N4096 | causal | matched | 5.116 | 5.015 | -1.98% | 0.38% | 0.12% |
| D128 B8 H16/16 N4096 | causal | stale | 4.507 | 4.406 | -2.23% | 0.08% | 0.04% |
| D128 B8 H16/16 N4096 | full | stale | 7.474 | 7.326 | -1.98% | 0.71% | 0.41% |
| D128 B16 H16/4 N2048 | causal | matched | 3.192 | 3.143 | -1.56% | 0.57% | 0.60% |
| D128 B16 H16/4 N2048 | causal | stale | 2.867 | 2.805 | -2.16% | 0.43% | 0.23% |
| D128 B16 H16/4 N2048 | full | stale | 4.301 | 4.188 | -2.62% | 0.43% | 0.10% |
| D128 B16 H16/16 N2048 | causal | matched | 3.510 | 3.471 | -1.13% | 2.57% | 3.03% |
| D128 B16 H16/16 N2048 | causal | stale | 3.045 | 2.989 | -1.82% | 0.39% | 0.31% |
| D128 B16 H16/16 N2048 | full | stale | 4.486 | 4.386 | -2.24% | 0.34% | 0.12% |
| D256 B1 H8/2 N32768 | causal | matched | 30.388 | 30.174 | -0.70% | 0.68% | 0.23% |
| D256 B1 H8/2 N32768 | causal | stale | 25.721 | 25.706 | -0.06% | 0.15% | 0.13% |
| D256 B1 H8/2 N32768 | full | stale | 51.205 | 51.043 | -0.32% | 0.60% | 0.48% |
| D256 B1 H8/8 N32768 | causal | matched | 30.393 | 30.467 | +0.24% | 1.08% | 0.90% |
| D256 B1 H8/8 N32768 | causal | stale | 25.998 | 25.987 | -0.04% | 1.05% | 0.72% |
| D256 B1 H8/8 N32768 | full | stale | 51.211 | 51.110 | -0.20% | 0.40% | 0.52% |
| D256 B1 H12/4 N12288 | causal | matched | 6.829 | 6.884 | +0.80% | 0.72% | 3.73% |
| D256 B1 H12/4 N12288 | causal | stale | 5.947 | 5.903 | -0.72% | 0.11% | 0.64% |
| D256 B1 H12/4 N12288 | full | stale | 11.177 | 11.110 | -0.60% | 1.42% | 0.65% |
| D256 B2 H8/2 N16384 | causal | matched | 15.874 | 15.750 | -0.78% | 2.52% | 1.86% |
| D256 B2 H8/2 N16384 | causal | stale | 13.452 | 13.515 | +0.47% | 0.55% | 0.47% |
| D256 B2 H8/2 N16384 | full | stale | 25.990 | 25.912 | -0.30% | 0.80% | 0.64% |
| D256 B2 H8/8 N16384 | causal | matched | 15.975 | 15.888 | -0.54% | 0.52% | 0.33% |
| D256 B2 H8/8 N16384 | causal | stale | 13.684 | 13.613 | -0.52% | 0.84% | 0.65% |
| D256 B2 H8/8 N16384 | full | stale | 26.089 | 26.070 | -0.08% | 1.18% | 0.18% |
| D256 B2 H16/1 N5120 | causal | matched | 3.391 | 3.387 | -0.11% | 0.13% | 0.17% |
| D256 B2 H16/1 N5120 | causal | stale | 3.010 | 3.012 | +0.08% | 0.66% | 0.18% |
| D256 B2 H16/1 N5120 | full | stale | 5.414 | 5.413 | -0.01% | 0.12% | 0.10% |
| D256 B3 H6/2 N6144 | causal | matched | 2.967 | 2.968 | +0.03% | 0.19% | 0.20% |
| D256 B3 H6/2 N6144 | causal | stale | 2.637 | 2.637 | -0.03% | 0.11% | 0.03% |
| D256 B3 H6/2 N6144 | full | stale | 4.669 | 4.664 | -0.10% | 0.53% | 0.07% |
| D256 B4 H8/2 N8192 | causal | matched | 8.457 | 8.400 | -0.67% | 0.85% | 2.36% |
| D256 B4 H8/2 N8192 | causal | stale | 7.332 | 7.363 | +0.42% | 0.73% | 1.20% |
| D256 B4 H8/2 N8192 | full | stale | 13.574 | 13.468 | -0.78% | 0.70% | 0.36% |
| D256 B4 H8/8 N8192 | causal | matched | 8.622 | 8.717 | +1.11% | 2.77% | 1.54% |
| D256 B4 H8/8 N8192 | causal | stale | 7.510 | 7.525 | +0.21% | 0.93% | 0.37% |
| D256 B4 H8/8 N8192 | full | stale | 13.699 | 13.625 | -0.54% | 0.92% | 1.33% |
| D256 B8 H8/2 N4096 | causal | matched | 4.796 | 4.800 | +0.08% | 1.27% | 0.56% |
| D256 B8 H8/2 N4096 | causal | stale | 4.315 | 4.318 | +0.08% | 0.10% | 0.25% |
| D256 B8 H8/2 N4096 | full | stale | 7.345 | 7.345 | -0.00% | 0.71% | 0.00% |
| D256 B8 H8/8 N4096 | causal | matched | 4.996 | 4.989 | -0.15% | 1.26% | 0.92% |
| D256 B8 H8/8 N4096 | causal | stale | 4.486 | 4.488 | +0.06% | 0.13% | 0.08% |
| D256 B8 H8/8 N4096 | full | stale | 7.536 | 7.516 | -0.26% | 0.36% | 0.76% |
| D256 B16 H8/2 N2048 | causal | matched | 3.097 | 3.100 | +0.11% | 0.14% | 0.13% |
| D256 B16 H8/2 N2048 | causal | stale | 2.828 | 2.829 | +0.03% | 0.22% | 0.11% |
| D256 B16 H8/2 N2048 | full | stale | 4.330 | 4.328 | -0.04% | 0.11% | 0.18% |
| D256 B16 H8/8 N2048 | causal | matched | 3.293 | 3.294 | +0.01% | 0.29% | 0.31% |
| D256 B16 H8/8 N2048 | causal | stale | 2.993 | 2.991 | -0.05% | 0.09% | 0.05% |
| D256 B16 H8/8 N2048 | full | stale | 4.437 | 4.439 | +0.05% | 0.36% | 0.06% |

Worst single-shape fwd+bwd change (B vs A): +1.11%
