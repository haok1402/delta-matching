### acceptance (4 configurations, model sol4-v2, calibration f20f440e2728; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 178.814 | 75.913 | 29.710 | 71.871 | 36.781 | 75.913 | tensor | 42.5% | 102.901 |
| fwd | 65.840 | 37.938 | 32.675 | 32.621 | 0.562 | 37.938 | tensor | 57.6% | 27.902 |
| pass0 | 54.491 | 38.046 | 28.028 | 29.452 | 0.400 | 38.046 | tensor | 69.8% | 16.445 |
| pass2 | 37.531 | 18.978 | 0.000 | 3.705 | 35.740 | 35.740 | bytes | 95.2% | 1.791 |
| glue | 3.158 | 0.000 | 0.000 | n/a | 2.383 | 2.383 | bytes | 75.5% | 0.775 |
| prod_fwd | 1.143 | 0.000 | 0.000 | n/a | 0.755 | 0.755 | bytes | 66.1% | 0.388 |
| prod_bwd | 0.627 | 0.000 | 0.000 | n/a | 0.501 | 0.501 | bytes | 80.0% | 0.126 |
| twin | 0.710 | 0.000 | 0.000 | n/a | 0.668 | 0.668 | bytes | 94.1% | 0.042 |
