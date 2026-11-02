# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_raw.json` (job 21171649, pool0-01619); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 518.4 +- 7.5 | 482.2 +- 9.0 | 491.7 +- 6.1 | 1.061 | 3.916 |
| H16/dm_base | 529.5 +- 7.5 | 449.7 +- 2.6 | 469.9 +- 3.0 | 1.039 | 4.096 |
| H16/fa3_bf16 | 693.8 +- 0.3 | 516.2 +- 10.4 | 556.7 +- 8.6 | 0.792 | 3.459 |
| H16/cudnn_bf16 | 596.4 +- 1.4 | 478.3 +- 8.1 | 506.8 +- 6.2 | 0.922 | 3.799 |
| H16/te_fp8 | not run: ModuleNotFoundError: No module named 'transformer_engine' | | | | |
| H32/dm_final | 564.6 +- 7.0 | 476.3 +- 1.8 | 498.5 +- 0.2 | 1.949 | 7.720 |
| H32/dm_base | 576.5 +- 3.4 | 454.6 +- 1.6 | 483.8 +- 0.9 | 1.908 | 7.955 |
| H32/fa3_bf16 | 715.5 +- 4.0 | 524.5 +- 3.2 | 567.8 +- 2.5 | 1.537 | 6.778 |
| H32/cudnn_bf16 | 624.0 +- 2.6 | 467.2 +- 1.7 | 503.3 +- 1.1 | 1.762 | 7.646 |
| H32/te_fp8 | not run: ModuleNotFoundError: No module named 'transformer_engine' | | | | |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.88x, dm_base 0.84x, fa3_bf16 1.00x, cudnn_bf16 0.91x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.88x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
