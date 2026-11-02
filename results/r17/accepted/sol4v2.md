### acceptance (4 configurations, model sol4-v2, calibration 17c81d268809; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 187.658 | 74.176 | 31.001 | 74.768 | 36.596 | 74.780 | issue | 39.8% | 112.878 |
| fwd | 75.842 | 37.176 | 33.804 | 36.413 | 0.559 | 37.176 | tensor | 49.0% | 38.666 |
| pass0 | 59.865 | 37.088 | 29.178 | 30.090 | 0.398 | 37.088 | tensor | 62.0% | 22.777 |
| pass2 | 37.721 | 18.544 | 0.000 | 3.866 | 35.561 | 35.561 | bytes | 94.3% | 2.161 |
| glue | 3.169 | 0.000 | 0.000 | n/a | 2.371 | 2.371 | bytes | 74.8% | 0.798 |
| prod_fwd | 1.146 | 0.000 | 0.000 | n/a | 0.751 | 0.751 | bytes | 65.6% | 0.395 |
| prod_bwd | 0.626 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.7% | 0.127 |
| twin | 0.695 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 95.7% | 0.030 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.957 ms; our forward / FA3 fp8 = 1.3086
