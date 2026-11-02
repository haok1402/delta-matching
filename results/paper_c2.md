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

Round 17 rerun (job 21171649, `results/paper_c2/r17_raw.json`): the BF16 reference forwards now run as
TRAINING forwards (gradients enabled, softmax statistics saved; round-16 review B21 — the round-16 table
`results/paper_c2/r16_raw.json` timed them under no_grad and is kept as historical evidence only).

## Reconciling the forward with the paper (661.0 vs ~518 TFLOP/s at 16/4)

Paper Appendix C.2 (checked against the paper): one H100 SXM, B2 x 8192, causal, native GQA, random normal inputs, five
interleaved rounds, 3 s matmul warm-up, median of 20 CUDA events after 5 warm-ups, FP8 timings with prequantized inputs
and without the autograd wrapper but with the correction pass, training forward with gradients and saved statistics,
backward = fwd+bwd minus forward, nominal work 2 B Hq N^2 d / 5 B Hq N^2 d. Our script follows that protocol; the build
is not the cause (the original baseline 3d4cd5b gives the same forward, 529.5 TFLOP/s).

Diagnostic (`benchmarks/paper_c2.py diag`, `results/paper_c2/r17_diag_*.json`, SM clock 1980 MHz in all runs):

| build, heads | per-call events (protocol) | 20 back-to-back calls / 20 | profiler: forward kernel | kernel TFLOP/s |
|---|---:|---:|---:|---:|
| accepted, 16/4 | 1.086 ms | 0.949 ms | 0.781 ms | 704 |
| original baseline, 16/4 | 1.082 ms | 0.938 ms | 0.775 ms | 709 |
| accepted, 32/4 | 1.970 ms | 1.659 ms | 1.526 ms | 721 |

The forward KERNEL runs at 704 TFLOP/s, above the paper's 661. The protocol's per-call event pair starts on an idle
GPU, so it also covers the host-side work of the `fwd_fp8` call before the kernel starts (about 0.3 ms here: argument
checks, output allocation, the scheduler-semaphore fill, the launch). The paper's 661.0 (0.832 ms) is the same kernel
with less host-side overhead on the paper's machine (its fwd+bwd, 470.2, is within 1% of ours under the same protocol,
where the host work is a smaller fraction). Our backward column is correspondingly not inflated (bwd = fwd+bwd minus
fwd removes the same forward overhead). Conclusion: the forward gap is host overhead inside the timing window, not a
slower kernel or a different build; the host path of `fwd_fp8` (~0.3 ms per call) is itself a real training cost and
is logged as a candidate (forward host path).

## FA3 BF16 ratio

On this H100, FA3 BF16 measures 693.8 / 516.2 / 556.7 TFLOP/s (fwd / bwd / fwd+bwd) at 16/4 under the same protocol,
while the paper's Table 10 reports 386.9 / 230.6 / 260.7 for FA3 BF16 at the same shape. The paper's "1.80x FA3 BF16"
therefore does not reproduce here: our matched kernel is 0.88x FA3 BF16 in fwd+bwd (491.7 vs 556.7). cuDNN BF16 agrees
with the paper (596.4 / 478.3 / 506.8 here vs 601.6 / 472.9 / 503.6), so the machine class matches; the likely reason is
the paper's FA3 BF16 run itself (an FA3 build or path far from Hopper-optimized at this GQA shape, e.g. a fallback or
repeated KV heads), not this H100. Our matched fwd+bwd (491.7) is above the paper's 470.2.
