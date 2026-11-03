# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `/home/hcai/workspace/haok1402/kda-results/paper_c2/r24_stack_raw.json` (job 21211208, pool0-01258); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 613.5 +- 7.7 | 498.1 +- 2.9 | 526.3 +- 2.1 | 0.897 | 3.656 |
| H16/dm_base | 514.0 +- 5.3 | 448.8 +- 2.9 | 465.6 +- 1.6 | 1.070 | 4.133 |
| H16/fa3_bf16 | 702.6 +- 0.3 | 505.1 +- 2.1 | 549.2 +- 1.8 | 0.782 | 3.504 |
| H16/cudnn_bf16 | 595.7 +- 2.6 | 492.9 +- 9.4 | 518.3 +- 7.5 | 0.923 | 3.715 |
| H32/dm_final | 663.0 +- 8.7 | 519.3 +- 2.9 | 553.4 +- 0.6 | 1.660 | 6.953 |
| H32/dm_base | 574.5 +- 6.6 | 451.3 +- 1.5 | 480.7 +- 0.7 | 1.915 | 8.005 |
| H32/fa3_bf16 | 729.7 +- 1.3 | 523.5 +- 1.3 | 569.4 +- 1.2 | 1.507 | 6.758 |
| H32/cudnn_bf16 | 623.3 +- 4.2 | 474.1 +- 1.3 | 508.9 +- 0.9 | 1.764 | 7.562 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.96x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.97x, dm_base 0.84x, fa3_bf16 1.00x, cudnn_bf16 0.89x
