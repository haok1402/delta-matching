### acceptance (4 configurations, model sol4-v2, calibration c27f4b16a271; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 177.498 | 80.026 | 27.962 | 67.643 | 36.752 | 80.026 | tensor | 45.1% | 97.471 |
| fwd | 76.607 | 40.108 | 30.490 | 32.843 | 0.562 | 40.108 | tensor | 52.4% | 36.499 |
| pass0 | 57.469 | 40.108 | 26.380 | 28.595 | 0.400 | 40.108 | tensor | 69.8% | 17.361 |
| pass2 | 37.704 | 20.007 | 0.000 | 3.487 | 35.712 | 35.712 | bytes | 94.7% | 1.992 |
| glue | 3.171 | 0.000 | 0.000 | n/a | 2.381 | 2.381 | bytes | 75.1% | 0.790 |
| prod_fwd | 1.155 | 0.000 | 0.000 | n/a | 0.755 | 0.755 | bytes | 65.3% | 0.401 |
| prod_bwd | 0.627 | 0.000 | 0.000 | n/a | 0.501 | 0.501 | bytes | 79.9% | 0.126 |
| twin | 0.709 | 0.000 | 0.000 | n/a | 0.668 | 0.668 | bytes | 94.2% | 0.041 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.656 ms; our forward / FA3 fp8 = 1.3287
