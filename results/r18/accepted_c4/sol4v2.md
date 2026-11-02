### acceptance (4 configurations, model sol4-v2, calibration 1f7674a4409c; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pass1 | 172.733 | 74.992 | 27.009 | 58.287 | 36.745 | 74.992 | tensor | 43.4% | 120.955 | 70.0% | 97.741 |
| fwd | 66.972 | 37.478 | 29.705 | 29.655 | 0.562 | 37.478 | tensor | 56.0% | 53.617 | 80.1% | 29.494 |
| pass0 | 50.358 | 37.585 | 25.480 | 20.603 | 0.400 | 37.585 | tensor | 74.6% | 50.449 | 100.2% | 12.773 |
| pass2 | 37.881 | 18.748 | 0.000 | 3.368 | 35.705 | 35.705 | bytes | 94.3% | 35.705 | 94.3% | 2.176 |
| glue | 3.175 | 0.000 | 0.000 | n/a | 2.381 | 2.381 | bytes | 75.0% | 2.381 | 75.0% | 0.794 |
| prod_fwd | 1.164 | 0.000 | 0.000 | n/a | 0.754 | 0.754 | bytes | 64.8% | 0.754 | 64.8% | 0.409 |
| prod_bwd | 0.631 | 0.000 | 0.000 | n/a | 0.501 | 0.501 | bytes | 79.3% | 0.501 | 79.3% | 0.130 |
| twin | 0.710 | 0.000 | 0.000 | n/a | 0.667 | 0.667 | bytes | 94.0% | 0.667 | 94.0% | 0.043 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.644 ms; our forward / FA3 fp8 = 1.1618
