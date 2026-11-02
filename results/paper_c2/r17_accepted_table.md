# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_accepted_raw.json` (job 21172815, pool0-01696); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 518.8 +- 7.4 | 467.7 +- 4.8 | 481.1 +- 2.8 | 1.061 | 4.000 |
| H16/dm_base | 509.9 +- 3.7 | 444.8 +- 2.5 | 461.6 +- 1.1 | 1.078 | 4.169 |
| H16/fa3_bf16 | 693.9 +- 0.2 | 505.6 +- 6.5 | 548.0 +- 5.4 | 0.792 | 3.512 |
| H16/cudnn_bf16 | 590.7 +- 3.3 | 471.4 +- 5.9 | 500.2 +- 4.8 | 0.931 | 3.848 |
| H32/dm_final | 576.4 +- 4.6 | 475.1 +- 2.4 | 500.1 +- 1.2 | 1.908 | 7.695 |
| H32/dm_base | 562.4 +- 6.7 | 457.0 +- 2.6 | 482.8 +- 1.5 | 1.956 | 7.971 |
| H32/fa3_bf16 | 719.9 +- 0.4 | 523.2 +- 0.9 | 567.5 +- 0.8 | 1.527 | 6.781 |
| H32/cudnn_bf16 | 619.8 +- 3.8 | 465.1 +- 2.0 | 500.8 +- 1.2 | 1.774 | 7.685 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.88x, dm_base 0.84x, fa3_bf16 1.00x, cudnn_bf16 0.91x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.88x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.88x
