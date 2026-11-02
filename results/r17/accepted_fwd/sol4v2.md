### acceptance (4 configurations, model sol4-v2, calibration 833d75563d6e; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 181.442 | 79.729 | 27.009 | 65.337 | 36.793 | 79.729 | tensor | 43.9% | 101.714 |
| fwd | 67.527 | 39.845 | 29.705 | 29.655 | 0.562 | 39.845 | tensor | 59.0% | 27.681 |
| pass0 | 58.764 | 39.958 | 25.480 | 27.620 | 0.400 | 39.958 | tensor | 68.0% | 18.805 |
| pass2 | 37.815 | 19.932 | 0.000 | 3.368 | 35.752 | 35.752 | bytes | 94.5% | 2.063 |
| glue | 3.173 | 0.000 | 0.000 | n/a | 2.384 | 2.384 | bytes | 75.1% | 0.789 |
| prod_fwd | 1.159 | 0.000 | 0.000 | n/a | 0.755 | 0.755 | bytes | 65.2% | 0.403 |
| prod_bwd | 0.632 | 0.000 | 0.000 | n/a | 0.501 | 0.501 | bytes | 79.3% | 0.131 |
| twin | 0.712 | 0.000 | 0.000 | n/a | 0.668 | 0.668 | bytes | 93.9% | 0.043 |

FA3 fp8 forward kernel (per-configuration medians, summed): 58.072 ms; our forward / FA3 fp8 = 1.1628
