# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_p1_raw.json` (job 21175599, pool0-00565); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 516.9 +- 3.7 | 498.6 +- 2.3 | 503.7 +- 1.4 | 1.064 | 3.820 |
| H16/dm_base | 517.2 +- 7.8 | 448.0 +- 2.8 | 465.6 +- 0.8 | 1.064 | 4.132 |
| H16/fa3_bf16 | 696.9 +- 0.3 | 511.1 +- 1.8 | 553.2 +- 1.5 | 0.789 | 3.478 |
| H16/cudnn_bf16 | 590.7 +- 4.6 | 487.7 +- 2.6 | 513.2 +- 1.8 | 0.931 | 3.750 |
| H32/dm_final | 569.7 +- 5.6 | 501.1 +- 1.9 | 518.9 +- 0.2 | 1.931 | 7.416 |
| H32/dm_base | 580.6 +- 6.4 | 456.5 +- 1.8 | 486.1 +- 0.3 | 1.895 | 7.917 |
| H32/fa3_bf16 | 723.0 +- 1.0 | 525.4 +- 1.8 | 569.9 +- 1.4 | 1.521 | 6.753 |
| H32/cudnn_bf16 | 621.1 +- 3.4 | 475.1 +- 1.5 | 509.3 +- 0.8 | 1.770 | 7.556 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.91x, dm_base 0.84x, fa3_bf16 1.00x, cudnn_bf16 0.93x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.91x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
