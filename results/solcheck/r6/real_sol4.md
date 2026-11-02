### acceptance (4 configurations, model sol4-v2, calibration 5b2b95becd45; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 179.765 | 76.050 | 27.009 | 65.337 | 36.632 | 76.050 | tensor | 42.3% | 103.715 |
| fwd | 66.142 | 38.007 | 29.705 | 29.655 | 0.560 | 38.007 | tensor | 57.5% | 28.136 |
| pass0 | 54.652 | 38.115 | 25.480 | 26.774 | 0.399 | 38.115 | tensor | 69.7% | 16.537 |
| pass2 | 37.673 | 19.012 | 0.000 | 3.368 | 35.596 | 35.596 | bytes | 94.5% | 2.077 |
| glue | 3.165 | 0.000 | 0.000 | n/a | 2.373 | 2.373 | bytes | 75.0% | 0.791 |
| prod_fwd | 1.149 | 0.000 | 0.000 | n/a | 0.752 | 0.752 | bytes | 65.5% | 0.397 |
| prod_bwd | 0.628 | 0.000 | 0.000 | n/a | 0.499 | 0.499 | bytes | 79.5% | 0.129 |
| twin | 0.696 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 95.6% | 0.031 |
