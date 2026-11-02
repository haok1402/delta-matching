# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r19_fwd_raw.json` (job 21196592, pool0-01668); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 621.5 +- 7.9 | 500.3 +- 3.7 | 529.7 +- 3.0 | 0.885 | 3.633 |
| H16/dm_base | 522.6 +- 7.1 | 448.7 +- 2.0 | 467.5 +- 0.7 | 1.053 | 4.116 |
| H16/fa3_bf16 | 703.9 +- 0.2 | 508.3 +- 3.8 | 552.1 +- 3.2 | 0.781 | 3.486 |
| H16/cudnn_bf16 | 598.7 +- 2.1 | 494.8 +- 6.5 | 520.6 +- 5.4 | 0.918 | 3.698 |
| H32/dm_final | 651.3 +- 7.1 | 526.7 +- 3.0 | 557.1 +- 1.1 | 1.689 | 6.908 |
| H32/dm_base | 580.3 +- 5.6 | 455.7 +- 1.5 | 485.4 +- 0.2 | 1.895 | 7.928 |
| H32/fa3_bf16 | 731.8 +- 0.9 | 529.7 +- 1.7 | 575.1 +- 1.5 | 1.502 | 6.692 |
| H32/cudnn_bf16 | 624.1 +- 3.9 | 476.3 +- 1.5 | 510.9 +- 0.9 | 1.762 | 7.533 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.96x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.97x, dm_base 0.84x, fa3_bf16 1.00x, cudnn_bf16 0.89x
