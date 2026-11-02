### acceptance (4 configurations, model sol4-v2, calibration 65e9614c3eac; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pass1 | 171.766 | 76.611 | 27.009 | 58.287 | 36.651 | 76.611 | tensor | 44.6% | 123.566 | 71.9% | 95.156 |
| fwd | 63.919 | 38.287 | 29.705 | 28.814 | 0.560 | 38.287 | tensor | 59.9% | 54.774 | 85.7% | 25.632 |
| pass0 | 50.779 | 38.396 | 25.480 | 20.603 | 0.399 | 38.396 | tensor | 75.6% | 51.538 | 101.5% | 12.383 |
| pass2 | 37.750 | 19.153 | 0.000 | 3.368 | 35.614 | 35.614 | bytes | 94.3% | 35.614 | 94.3% | 2.136 |
| glue | 3.171 | 0.000 | 0.000 | n/a | 2.375 | 2.375 | bytes | 74.9% | 2.375 | 74.9% | 0.797 |
| prod_fwd | 1.157 | 0.000 | 0.000 | n/a | 0.753 | 0.753 | bytes | 65.0% | 0.753 | 65.0% | 0.405 |
| prod_bwd | 0.632 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.0% | 0.499 | 79.0% | 0.133 |
| twin | 0.697 | 0.000 | 0.000 | n/a | 0.666 | 0.666 | bytes | 95.6% | 0.666 | 95.6% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.663 ms; our forward / FA3 fp8 = 1.1085
