# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_c3_raw.json` (job 21174139, pool0-00648); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 517.6 +- 6.7 | 482.4 +- 5.9 | 491.9 +- 5.3 | 1.063 | 3.914 |
| H16/dm_base | 523.0 +- 5.3 | 445.9 +- 6.6 | 465.4 +- 5.7 | 1.052 | 4.136 |
| H16/fa3_bf16 | 697.0 +- 0.6 | 502.3 +- 8.9 | 545.7 +- 7.5 | 0.789 | 3.529 |
| H16/cudnn_bf16 | 591.1 +- 5.5 | 486.2 +- 9.5 | 511.9 +- 6.2 | 0.930 | 3.761 |
| H32/dm_final | 555.1 +- 2.6 | 488.7 +- 1.2 | 506.0 +- 1.3 | 1.981 | 7.606 |
| H32/dm_base | 567.0 +- 4.2 | 455.9 +- 1.0 | 482.9 +- 0.4 | 1.940 | 7.969 |
| H32/fa3_bf16 | 725.7 +- 0.7 | 524.6 +- 1.9 | 569.7 +- 1.5 | 1.515 | 6.755 |
| H32/cudnn_bf16 | 627.7 +- 0.6 | 467.4 +- 2.9 | 504.2 +- 2.4 | 1.752 | 7.634 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.90x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.89x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.88x
