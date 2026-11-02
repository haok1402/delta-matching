# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r18_c4_raw.json` (job 21191934, pool0-01626); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 568.9 +- 1.4 | 505.3 +- 6.9 | 521.9 +- 5.0 | 0.966 | 3.688 |
| H16/dm_base | 513.5 +- 8.9 | 440.3 +- 4.0 | 458.9 +- 3.5 | 1.072 | 4.194 |
| H16/fa3_bf16 | 691.0 +- 0.3 | 498.9 +- 1.7 | 541.9 +- 1.4 | 0.796 | 3.551 |
| H16/cudnn_bf16 | 582.7 +- 5.3 | 486.2 +- 8.4 | 510.2 +- 5.9 | 0.944 | 3.774 |
| H32/dm_final | 628.0 +- 6.9 | 522.3 +- 2.0 | 548.6 +- 0.7 | 1.752 | 7.015 |
| H32/dm_base | 573.4 +- 5.5 | 450.7 +- 1.8 | 480.0 +- 0.9 | 1.918 | 8.017 |
| H32/fa3_bf16 | 716.2 +- 0.4 | 519.8 +- 1.3 | 564.0 +- 1.1 | 1.535 | 6.824 |
| H32/cudnn_bf16 | 617.5 +- 2.5 | 466.4 +- 0.9 | 501.4 +- 0.7 | 1.781 | 7.675 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.96x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.97x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
