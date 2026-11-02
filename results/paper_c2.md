# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r16_raw.json` (job 21170691, pool0-01565); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 506.7 +- 3.0 | 458.9 +- 1.6 | 471.6 +- 0.6 | 1.085 | 4.080 |
| H16/dm_base | 518.0 +- 5.1 | 443.4 +- 2.5 | 462.4 +- 1.4 | 1.062 | 4.161 |
| H16/fa3_bf16 | 696.4 +- 0.5 | 506.6 +- 11.7 | 549.2 +- 9.8 | 0.789 | 3.508 |
| H16/cudnn_bf16 | 582.6 +- 5.7 | 475.3 +- 1.2 | 501.7 +- 2.1 | 0.944 | 3.835 |
| H16/te_fp8 | not run: ModuleNotFoundError: No module named 'transformer_engine' | | | | |
| H32/dm_final | 564.4 +- 5.6 | 467.7 +- 1.2 | 491.7 +- 0.8 | 1.949 | 7.826 |
| H32/dm_base | 563.6 +- 5.6 | 451.9 +- 2.2 | 478.9 +- 0.8 | 1.952 | 8.035 |
| H32/fa3_bf16 | 723.7 +- 0.6 | 521.2 +- 1.5 | 566.5 +- 1.3 | 1.519 | 6.794 |
| H32/cudnn_bf16 | 616.2 +- 4.1 | 470.1 +- 1.7 | 504.2 +- 0.8 | 1.785 | 7.633 |
| H32/te_fp8 | not run: ModuleNotFoundError: No module named 'transformer_engine' | | | | |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.86x, dm_base 0.84x, fa3_bf16 1.00x, cudnn_bf16 0.91x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.87x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
