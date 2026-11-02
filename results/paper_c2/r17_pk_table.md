# Paper protocol (Appendix C.2): B2 N8192 Hq 16 / 32, Hk4 D128 causal

Raw: `results/paper_c2/r17_pk_raw.json` (job 21181967, pool0-01791); 5 interleaved rounds, 3 s matmul clock warm-up per benchmark, median of 20 CUDA events after 5 warm-ups; TFLOP/s from nominal work (fwd 2*B*Hq*N^2*d, bwd 5*B*Hq*N^2*d); mean +- SE over rounds; bwd = fwd+bwd - fwd. delta-matching: prequantized inputs, no autograd wrapper (GQA dK/dV fold excluded), matched-delta correction pass included.

| heads / arm | fwd TFLOP/s | bwd TFLOP/s | fwd+bwd TFLOP/s | fwd ms | fwd+bwd ms |
|---|---:|---:|---:|---:|---:|
| H16/dm_final | 582.8 +- 5.4 | 486.4 +- 2.6 | 510.5 +- 1.6 | 0.944 | 3.769 |
| H16/dm_base | 516.4 +- 7.5 | 442.3 +- 3.8 | 461.0 +- 1.8 | 1.065 | 4.174 |
| H16/fa3_bf16 | 697.0 +- 0.7 | 499.7 +- 3.3 | 543.6 +- 2.8 | 0.789 | 3.540 |
| H16/cudnn_bf16 | 590.3 +- 3.8 | 483.5 +- 9.3 | 509.6 +- 6.7 | 0.931 | 3.778 |
| H32/dm_final | 639.0 +- 6.1 | 500.3 +- 1.4 | 533.3 +- 1.4 | 1.721 | 7.216 |
| H32/dm_base | 564.4 +- 6.8 | 452.6 +- 3.6 | 479.6 +- 1.6 | 1.949 | 8.024 |
| H32/fa3_bf16 | 723.1 +- 0.7 | 519.0 +- 2.3 | 564.5 +- 1.9 | 1.521 | 6.817 |
| H32/cudnn_bf16 | 614.7 +- 5.2 | 468.8 +- 2.3 | 502.8 +- 1.1 | 1.789 | 7.654 |
| H16 / paper (matched) | 661.0 | 421.5 | 470.2 | | |

H16: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.94x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.94x

H32: fwd+bwd TFLOP/s relative to FA3 BF16: dm_final 0.94x, dm_base 0.85x, fa3_bf16 1.00x, cudnn_bf16 0.89x
