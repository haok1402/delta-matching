### acceptance (4 configurations, model sol4-v2, calibration f7e7d2cb60c9; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pass1 | 167.822 | 79.264 | 27.009 | 58.287 | 36.631 | 79.264 | tensor | 47.2% | 127.845 | 76.2% | 88.558 |
| fwd | 64.742 | 39.613 | 29.705 | 28.814 | 0.560 | 39.613 | tensor | 61.2% | 56.671 | 87.5% | 25.129 |
| pass0 | 46.745 | 39.725 | 25.480 | 15.328 | 0.399 | 39.725 | tensor | 85.0% | 53.323 | 114.1% | 7.019 |
| pass2 | 38.004 | 19.816 | 0.000 | 3.368 | 35.595 | 35.595 | bytes | 93.7% | 35.595 | 93.7% | 2.410 |
| glue | 3.186 | 0.000 | 0.000 | n/a | 2.373 | 2.373 | bytes | 74.5% | 2.373 | 74.5% | 0.813 |
| prod_fwd | 1.165 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 64.5% | 0.752 | 64.5% | 0.413 |
| prod_bwd | 0.635 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 78.6% | 0.499 | 78.6% | 0.136 |
| twin | 0.697 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 95.5% | 0.665 | 95.5% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.954 ms; our forward / FA3 fp8 = 1.1171
