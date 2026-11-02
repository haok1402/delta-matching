### target (8 configurations, model sol4-v2, calibration 78a0f4225064; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 26.042 | 10.932 | 3.683 | 8.883 | 6.701 | 10.932 | tensor | 42.0% | 15.110 |
| fwd | 11.106 | 5.578 | 3.867 | 4.165 | 0.805 | 5.578 | tensor | 50.2% | 5.528 |
| pass0 | 9.083 | 5.466 | 3.466 | 3.798 | 0.611 | 5.466 | tensor | 60.2% | 3.617 |
| glue | 1.653 | 0.000 | 0.000 | n/a | 0.665 | 0.665 | bytes | 40.2% | 0.988 |
| pass2 | 5.967 | 2.733 | 0.000 | 0.451 | 5.391 | 5.391 | bytes | 90.3% | 0.576 |
| prod_fwd | 1.902 | 0.000 | 0.000 | n/a | 1.510 | 1.510 | bytes | 79.4% | 0.392 |
| prod_bwd | 0.751 | 0.000 | 0.000 | n/a | 0.599 | 0.599 | bytes | 79.7% | 0.152 |
| twin | 0.828 | 0.000 | 0.000 | n/a | 0.798 | 0.798 | bytes | 96.4% | 0.030 |

FA3 fp8 forward kernel (per-configuration medians, summed): 8.795 ms; our forward / FA3 fp8 = 1.2628

### guard (8 configurations, model sol4-v2, calibration 78a0f4225064; issue / MUFU are loop-only lower bounds)

| kernel | measured ms | tensor | MUFU | issue | bytes | floor (max) | controlling | % of floor | waste ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| pass1 | 101.416 | 43.059 | 14.732 | 35.530 | 21.077 | 43.059 | tensor | 42.5% | 58.358 |
| fwd | 41.870 | 21.641 | 15.972 | 17.205 | 0.805 | 21.641 | tensor | 51.7% | 20.229 |
| pass0 | 34.409 | 21.529 | 13.865 | 15.192 | 0.611 | 21.529 | tensor | 62.6% | 12.880 |
| glue | 6.128 | 0.000 | 0.000 | n/a | 2.952 | 2.952 | bytes | 48.2% | 3.176 |
| pass2 | 21.437 | 10.765 | 0.000 | 1.832 | 19.767 | 19.767 | bytes | 92.2% | 1.669 |
| prod_fwd | 2.117 | 0.000 | 0.000 | n/a | 1.509 | 1.509 | bytes | 71.3% | 0.607 |
| prod_bwd | 0.770 | 0.000 | 0.000 | n/a | 0.599 | 0.599 | bytes | 77.8% | 0.171 |
| twin | 0.836 | 0.000 | 0.000 | n/a | 0.798 | 0.798 | bytes | 95.5% | 0.038 |

FA3 fp8 forward kernel (per-configuration medians, summed): 31.014 ms; our forward / FA3 fp8 = 1.3500
