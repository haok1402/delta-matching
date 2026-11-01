# Results (kda-sm90)

Per-kernel measurements of the delta_matching kernels on H100 SXM (cwpark, driver 535.216.03 + cuda-compat 580.196.07
forward compatibility, CUDA 13.0.2 toolkit, torch 2.12.0+cu130, Triton 3.7.0, FlashAttention-3 hopper at the pinned
commit 0251105 built with hdim 128/256 only).

- `baseline_main.json`, `baseline_holdout.json`: raw records of `benchmarks/kernels.py --grid main|holdout --fa3` on the
  unmodified kernels (main @ 3d4cd5b). `baseline.md`: `benchmarks/summarize.py` of the two.
- Main grid: D 128 (H 16) and D 256 (H 8), N 2k to 32k at 32k tokens per call (B = 32768 / N), MHA and GQA 4, causal
  matched, causal stale and full stale. Held-out grid (never tuned on): odd lengths (3k, 5k, 6k, 12k), batch 3, other
  head counts and GQA ratios (3, 4, 16).
- Method: device time per kernel from CUPTI (torch.profiler) over 10 fwd+bwd steps after 3 warm-ups; end-to-end fwd and
  fwd+bwd as the median of CUDA-event timings. The SM clock is sampled with nvidia-smi under a 4 s sustained load.
- Speed of light per kernel = max(matmul FLOPs / (132 SMs x 8192 FLOP/clk x sustained clock), compulsory bytes / 3.35 TB/s),
  with causal counting N (N + 1) / 2 visible pairs. The nominal 1979 TFLOPS corresponds to 1830 MHz; the GPU sustains
  about 1950 MHz under this load, a 2109 TFLOPS ceiling. The FLOP and byte model is `accounting()` in
  `benchmarks/kernels.py`.

## Tooling (round 0)

- `benchmarks/equivalence.py`: dump outputs + intermediates of one build per process (`PYTHONPATH=<build root>`), compare
  two dumps byte for byte. 72 cases (D 128/256, MHA/GQA4/GQA3/MQA, packed-QKV and (B,H,N,D) layouts, stress inputs,
  three modes) per route; the split routes (`--route kv|head|streams2`) cover the GQA/MQA cases.
- `benchmarks/ab.py`: `freeze` a revision's package + library; `run` alternates two builds over repetitions of
  `kernels.py` in fresh processes and reports per-kernel median sums, change and spread (`results/r0/`).
- `results/r0/`: self-tests: the baseline against itself is bitwise equal on every route and within 0.4% per kernel
  (worst single shape +0.86%, the per-shape noise floor); a one-bit perturbation and a slowed B are both detected.
- The modeled SOL (`kernels.accounting`) counts unique bytes with perfect overlap: an optimistic reference.
  `baseline.md` is regenerated with the corrected model (`summarize.py --remodel`).
