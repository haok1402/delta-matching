# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `/home/hcai/workspace/haok1402/kda-results/paper_c2/r21_clean_raw.json` (job 21206028, pool0-01736); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 534.7 +- 3.2 | 490.6 +- 1.6 | 502.4 +- 1.2 | 1.028 | 3.830 |
| H16/dm_base | 520.8 +- 7.4 | 446.2 +- 5.5 | 465.0 +- 2.7 | 1.056 | 4.138 |
| H16/fa3_bf16 | 697.0 +- 0.4 | 504.3 +- 4.4 | 547.5 +- 3.7 | 0.789 | 3.515 |
| H16/cudnn_bf16 | 590.2 +- 4.6 | 490.8 +- 10.1 | 515.3 +- 7.2 | 0.932 | 3.737 |
| H32/dm_final | 604.4 +- 2.1 | 499.7 +- 0.6 | 525.8 +- 0.8 | 1.819 | 7.320 |
| H32/dm_base | 575.9 +- 5.1 | 457.8 +- 1.8 | 486.2 +- 1.2 | 1.910 | 7.915 |
| H32/fa3_bf16 | 723.7 +- 0.4 | 530.9 +- 2.2 | 574.7 +- 1.8 | 1.519 | 6.697 |
| H32/cudnn_bf16 | 625.3 +- 0.4 | 474.7 +- 1.0 | 509.7 +- 0.9 | 1.758 | 7.550 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.92x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.91x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
