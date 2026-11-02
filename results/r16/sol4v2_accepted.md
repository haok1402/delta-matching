### acceptance (4 configurations, model sol4-v2, calibration 827e29e9ebcc; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 188.484 | 78.603 | 27.853 | 67.174 | 36.625 | 78.603 | tensor | 41.7% | 109.881 |
| fwd | 76.269 | 39.394 | 30.371 | 32.715 | 0.560 | 39.394 | tensor | 51.7% | 36.874 |
| pass0 | 63.701 | 39.302 | 26.214 | 28.723 | 0.399 | 39.302 | tensor | 61.7% | 24.400 |
| pass2 | 37.703 | 19.651 | 0.000 | 3.473 | 35.589 | 35.589 | bytes | 94.4% | 2.114 |
| glue | 3.172 | 0.000 | 0.000 | n/a | 2.373 | 2.373 | bytes | 74.8% | 0.799 |
| prod_fwd | 1.149 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 65.5% | 0.397 |
| prod_bwd | 0.628 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.5% | 0.128 |
| twin | 0.696 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 95.5% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 58.167 ms; our forward / FA3 fp8 = 1.3112
