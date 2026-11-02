job 21183777, calibrated fp8 GEMM 1437.3 TFLOP/s

| kernel | real % of sol4 | dep | free | ilv | spec | pp | best synthetic | real / best |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fwd | 57.6% | 64.9% | 69.2% | 65.8% | 69.7% | 66.8% | v_fwd_spec 69.7% | 0.827 |
| pass1 | 42.5% | 61.7% | 58.7% | 62.0% | 61.4% | - | v_pass1_ilv 62.0% | 0.685 |
| pass0 | 69.8% | 71.2% | 74.0% | 74.5% | 74.2% | - | v_pass0_ilv 74.5% | 0.937 |

job 21183778, calibrated fp8 GEMM 1434.7 TFLOP/s

| kernel | real % of sol4 | dep | free | ilv | spec | pp | best synthetic | real / best |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fwd | 57.5% | 64.5% | 68.8% | 65.9% | 70.1% | 66.7% | v_fwd_spec 70.1% | 0.820 |
| pass1 | 42.3% | 60.9% | 58.6% | 62.1% | 61.7% | - | v_pass1_ilv 62.1% | 0.681 |
| pass0 | 69.7% | 70.8% | 74.6% | 74.3% | 74.5% | - | syn_pass0_free 74.6% | 0.935 |

Mean over jobs:

| kernel | real % of sol4 | best synthetic % of sol4 | real / best |
|---|---:|---:|---:|
| fwd | 57.5% | 69.9% | 0.823 |
| pass1 | 42.4% | 62.0% | 0.683 |
| pass0 | 69.8% | 74.5% | 0.936 |
