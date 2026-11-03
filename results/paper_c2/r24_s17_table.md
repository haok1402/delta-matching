# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `/home/hcai/workspace/haok1402/kda-results/paper_c2/r24_s17_raw.json` (job 21213177, pool0-01804); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 601.1 +- 5.1 | 516.2 +- 3.1 | 537.9 +- 2.0 | 0.915 | 3.578 |
| H16/dm_base | 521.4 +- 2.8 | 445.3 +- 2.2 | 464.6 +- 2.0 | 1.055 | 4.141 |
| H16/fa3_bf16 | 690.3 +- 0.1 | 503.8 +- 1.9 | 545.9 +- 1.5 | 0.796 | 3.525 |
| H16/cudnn_bf16 | 587.8 +- 3.0 | 480.4 +- 12.1 | 506.6 +- 9.5 | 0.935 | 3.804 |
| H32/dm_final | 661.6 +- 2.6 | 534.2 +- 1.9 | 565.3 +- 1.1 | 1.662 | 6.808 |
| H32/dm_base | 563.2 +- 3.1 | 457.2 +- 2.1 | 483.2 +- 1.4 | 1.952 | 7.964 |
| H32/fa3_bf16 | 711.2 +- 2.5 | 523.8 +- 1.6 | 566.4 +- 1.3 | 1.546 | 6.794 |
| H32/cudnn_bf16 | 620.3 +- 0.4 | 466.8 +- 1.0 | 502.4 +- 0.8 | 1.772 | 7.661 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.99x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.93x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 1.00x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
