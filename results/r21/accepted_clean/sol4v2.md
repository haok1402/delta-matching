### acceptance (4 configurations, model sol4-v2, calibration 67d7a6011e3a; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pass1 | 176.506 | 75.968 | 27.009 | 64.543 | 36.643 | 75.968 | tensor | 43.0% | 122.530 | 69.4% | 100.538 |
| fwd | 73.846 | 38.074 | 29.450 | 31.452 | 0.560 | 38.074 | tensor | 51.6% | 54.469 | 73.8% | 35.772 |
| pass0 | 54.355 | 38.074 | 25.480 | 26.774 | 0.399 | 38.074 | tensor | 70.0% | 51.106 | 94.0% | 16.281 |
| pass2 | 37.692 | 18.992 | 0.000 | 3.368 | 35.607 | 35.607 | bytes | 94.5% | 35.607 | 94.5% | 2.086 |
| glue | 3.168 | 0.000 | 0.000 | n/a | 2.374 | 2.374 | bytes | 75.0% | 2.374 | 75.0% | 0.793 |
| prod_fwd | 1.152 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 65.3% | 0.752 | 65.3% | 0.399 |
| prod_bwd | 0.627 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.6% | 0.499 | 79.6% | 0.128 |
| twin | 0.697 | 0.000 | 0.000 | n/a | 0.666 | 0.666 | bytes | 95.5% | 0.666 | 95.5% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 57.462 ms; our forward / FA3 fp8 = 1.2851
