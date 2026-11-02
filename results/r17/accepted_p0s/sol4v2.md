### acceptance (4 configurations, model sol4-v2, calibration c845fa4a0c9d; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 181.298 | 77.416 | 27.009 | 65.337 | 36.630 | 77.416 | tensor | 42.7% | 103.882 |
| fwd | 66.470 | 38.690 | 29.705 | 29.655 | 0.560 | 38.690 | tensor | 58.2% | 27.780 |
| pass0 | 54.925 | 38.799 | 25.480 | 26.774 | 0.399 | 38.799 | tensor | 70.6% | 16.126 |
| pass2 | 37.730 | 19.354 | 0.000 | 3.368 | 35.593 | 35.593 | bytes | 94.3% | 2.137 |
| glue | 3.169 | 0.000 | 0.000 | n/a | 2.373 | 2.373 | bytes | 74.9% | 0.796 |
| prod_fwd | 1.154 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 65.2% | 0.402 |
| prod_bwd | 0.629 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.4% | 0.130 |
| twin | 0.697 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 95.5% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.999 ms; our forward / FA3 fp8 = 1.1460
