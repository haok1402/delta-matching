### acceptance (4 configurations, model sol4-v2, calibration 8742828f4b41; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | sol4c ms | % of sol4c | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pass1 | 179.293 | 73.665 | 27.009 | 62.259 | 36.628 | 73.665 | tensor | 41.1% | 118.815 | 66.3% | 105.628 |
| fwd | 67.089 | 36.815 | 29.705 | 29.655 | 0.560 | 36.815 | tensor | 54.9% | 52.668 | 78.5% | 30.274 |
| pass0 | 52.715 | 36.920 | 25.480 | 23.788 | 0.399 | 36.920 | tensor | 70.0% | 49.557 | 94.0% | 15.796 |
| pass2 | 37.807 | 18.416 | 0.000 | 3.368 | 35.592 | 35.592 | bytes | 94.1% | 35.592 | 94.1% | 2.216 |
| glue | 3.174 | 0.000 | 0.000 | n/a | 2.373 | 2.373 | bytes | 74.8% | 2.373 | 74.8% | 0.801 |
| prod_fwd | 1.159 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 64.9% | 0.752 | 64.9% | 0.407 |
| prod_bwd | 0.631 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.1% | 0.499 | 79.1% | 0.132 |
| twin | 0.697 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 95.5% | 0.665 | 95.5% | 0.031 |

FA3 fp8 forward kernel (per-configuration medians, summed): 58.112 ms; our forward / FA3 fp8 = 1.1545
