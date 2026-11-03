### acceptance (4 configurations, model sol4-v2, calibration 44115b7c4c4c; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pass1 | 172.545 | 79.551 | 30.602 | 66.042 | 36.752 | 79.551 | tensor | 46.1% | 128.308 | 74.4% | 92.994 |
| fwd | 64.047 | 39.756 | 33.657 | 32.647 | 0.562 | 39.756 | tensor | 62.1% | 56.876 | 88.8% | 24.290 |
| pass0 | 50.806 | 39.869 | 28.870 | 23.344 | 0.400 | 39.869 | tensor | 78.5% | 53.516 | 105.3% | 10.937 |
| pass2 | 37.818 | 19.888 | 0.000 | 3.816 | 35.712 | 35.712 | bytes | 94.4% | 35.712 | 94.4% | 2.106 |
| glue | 3.173 | 0.000 | 0.000 | n/a | 2.381 | 2.381 | bytes | 75.0% | 2.381 | 75.0% | 0.792 |
| prod_fwd | 1.158 | 0.000 | 0.000 | n/a | 0.755 | 0.755 | bytes | 65.1% | 0.755 | 65.1% | 0.404 |
| prod_bwd | 0.632 | 0.000 | 0.000 | n/a | 0.501 | 0.501 | bytes | 79.2% | 0.501 | 79.2% | 0.132 |
| twin | 0.710 | 0.000 | 0.000 | n/a | 0.668 | 0.668 | bytes | 94.1% | 0.668 | 94.1% | 0.042 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.799 ms; our forward / FA3 fp8 = 1.1081
