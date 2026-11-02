# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_p0s_raw.json` (job 21179953, pool0-01817); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 579.9 +- 6.1 | 481.2 +- 6.5 | 505.7 +- 5.5 | 0.949 | 3.807 |
| H16/dm_base | 522.8 +- 7.5 | 444.6 +- 3.1 | 464.3 +- 1.5 | 1.052 | 4.144 |
| H16/fa3_bf16 | 696.5 +- 0.2 | 501.0 +- 8.6 | 544.6 +- 7.2 | 0.789 | 3.536 |
| H16/cudnn_bf16 | 597.2 +- 1.8 | 476.7 +- 5.7 | 505.8 +- 4.9 | 0.921 | 3.805 |
| H32/dm_final | 620.8 +- 4.9 | 498.9 +- 2.7 | 528.5 +- 1.6 | 1.772 | 7.282 |
| H32/dm_base | 562.8 +- 7.0 | 460.9 +- 2.2 | 485.9 +- 0.3 | 1.955 | 7.920 |
| H32/fa3_bf16 | 725.1 +- 0.4 | 521.7 +- 1.6 | 567.1 +- 1.4 | 1.516 | 6.786 |
| H32/cudnn_bf16 | 625.5 +- 3.6 | 465.4 +- 1.2 | 502.1 +- 0.4 | 1.758 | 7.664 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.93x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.93x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.93x, dm_base 0.86x, fa3_bf16 1.00x, cudnn_bf16 0.89x
