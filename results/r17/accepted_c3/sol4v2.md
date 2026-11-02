### acceptance (4 configurations, model sol4-v2, calibration 641cda28f8c0; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 187.759 | 75.273 | 27.009 | 65.139 | 36.639 | 75.273 | tensor | 40.1% | 112.486 |
| fwd | 75.921 | 37.726 | 29.450 | 31.723 | 0.560 | 37.726 | tensor | 49.7% | 38.195 |
| pass0 | 57.267 | 37.726 | 25.480 | 27.620 | 0.399 | 37.726 | tensor | 65.9% | 19.541 |
| pass2 | 37.589 | 18.818 | 0.000 | 3.368 | 35.602 | 35.602 | bytes | 94.7% | 1.987 |
| glue | 3.168 | 0.000 | 0.000 | n/a | 2.374 | 2.374 | bytes | 74.9% | 0.794 |
| prod_fwd | 1.148 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 65.5% | 0.396 |
| prod_bwd | 0.626 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.8% | 0.126 |
| twin | 0.696 | 0.000 | 0.000 | n/a | 0.666 | 0.666 | bytes | 95.6% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.571 ms; our forward / FA3 fp8 = 1.3187
