# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_fwd_raw.json` (job 21178641, pool0-01821); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 581.6 +- 8.3 | 476.6 +- 2.6 | 502.4 +- 1.3 | 0.946 | 3.830 |
| H16/dm_base | 517.0 +- 6.9 | 446.4 +- 3.3 | 464.4 +- 1.2 | 1.064 | 4.144 |
| H16/fa3_bf16 | 688.3 +- 0.3 | 496.3 +- 3.9 | 539.2 +- 3.2 | 0.799 | 3.569 |
| H16/cudnn_bf16 | 588.2 +- 1.1 | 482.3 +- 10.0 | 508.2 +- 7.8 | 0.935 | 3.789 |
| H32/dm_final | 619.5 +- 6.5 | 492.5 +- 4.1 | 523.1 +- 2.2 | 1.776 | 7.358 |
| H32/dm_base | 569.5 +- 4.3 | 453.7 +- 1.9 | 481.6 +- 0.8 | 1.931 | 7.990 |
| H32/fa3_bf16 | 712.1 +- 1.6 | 518.5 +- 2.3 | 562.2 +- 2.1 | 1.544 | 6.846 |
| H32/cudnn_bf16 | 609.1 +- 4.0 | 466.6 +- 1.3 | 500.0 +- 0.7 | 1.805 | 7.696 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.93x, dm_base 0.86x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.93x, dm_base 0.86x, fa3_bf16 1.00x, cudnn_bf16 0.89x
