# A/B (acceptance run): grid holdout, 3 reps alternating A/B order, median per config

- A = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/base (library sha256 07fee0f25b8be7c9)
- B = /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching (library sha256 36f86c6183dc3c4f)

### Per-kernel sums, all configurations

| kernel | A ms (sum of medians) | B ms | change | A rep spread | B rep spread |
|---|---:|---:|---:|---:|---:|
| fwd | 17.224 | 17.244 | +0.12% | 0.23% | 0.35% |
| prod_fwd | 1.779 | 1.779 | -0.01% | 0.00% | 0.20% |
| prod_bwd | 1.199 | 1.200 | +0.07% | 0.10% | 0.09% |
| pass0 | 3.369 | 3.373 | +0.10% | 0.23% | 0.22% |
| pass1 | 46.477 | 46.238 | -0.51% | 0.23% | 0.25% |
| pass2 | 10.550 | 10.556 | +0.06% | 0.05% | 0.17% |
| twin | 0.805 | 0.805 | +0.04% | 0.26% | 0.04% |
| glue | 5.481 | 5.483 | +0.04% | 0.03% | 0.12% |
| e2e_fwd | 22.520 | 22.575 | +0.24% | 0.11% | 0.27% |
| e2e_fwdbwd | 87.148 | 86.971 | -0.20% | 0.29% | 0.16% |

### Shapes over the 1.0% fwd+bwd gate

- none

### Per shape (fwd+bwd, ms)

| shape | attn | mode | A median | B median | change | A spread | B spread |
|---|---|---|---:|---:|---:|---:|---:|
| D128 B1 H24/8 N12288 | causal | matched | 7.017 | 6.942 | -1.06% | 0.80% | 0.30% |
| D128 B1 H24/8 N12288 | causal | stale | 5.931 | 5.917 | -0.24% | 1.33% | 0.15% |
| D128 B1 H24/8 N12288 | full | stale | 11.189 | 11.147 | -0.38% | 1.09% | 0.46% |
| D128 B2 H32/32 N3072 | causal | matched | 1.712 | 1.697 | -0.87% | 0.72% | 0.78% |
| D128 B2 H32/32 N3072 | causal | stale | 1.500 | 1.496 | -0.25% | 1.29% | 0.31% |
| D128 B2 H32/32 N3072 | full | stale | 2.347 | 2.338 | -0.36% | 0.36% | 0.28% |
| D128 B3 H12/4 N6144 | causal | matched | 3.074 | 3.051 | -0.77% | 0.09% | 0.17% |
| D128 B3 H12/4 N6144 | causal | stale | 2.635 | 2.624 | -0.42% | 0.39% | 0.39% |
| D128 B3 H12/4 N6144 | full | stale | 4.684 | 4.648 | -0.77% | 0.16% | 0.59% |
| D256 B1 H12/4 N12288 | causal | matched | 7.022 | 6.986 | -0.51% | 1.12% | 2.24% |
| D256 B1 H12/4 N12288 | causal | stale | 6.103 | 6.105 | +0.04% | 0.10% | 0.72% |
| D256 B1 H12/4 N12288 | full | stale | 11.360 | 11.428 | +0.59% | 1.00% | 0.99% |
| D256 B2 H16/1 N5120 | causal | matched | 3.486 | 3.485 | -0.02% | 0.34% | 0.51% |
| D256 B2 H16/1 N5120 | causal | stale | 3.085 | 3.093 | +0.26% | 0.07% | 0.67% |
| D256 B2 H16/1 N5120 | full | stale | 5.512 | 5.518 | +0.10% | 0.10% | 0.78% |
| D256 B3 H6/2 N6144 | causal | matched | 3.045 | 3.046 | +0.04% | 0.82% | 0.58% |
| D256 B3 H6/2 N6144 | causal | stale | 2.716 | 2.717 | +0.03% | 0.50% | 0.73% |
| D256 B3 H6/2 N6144 | full | stale | 4.730 | 4.733 | +0.06% | 0.19% | 0.59% |

Worst single-shape fwd+bwd change (B vs A): +0.59%
