# sm100 (GB200) benchmark harness: Delta-Matching candidates vs FlashAttention-4, hd128

Snapshot of the working copy at ~/workspace/haok1402/sm100/bench (where it is run; scripts use those absolute paths).
- attnbench.py: the timing harness (method in its docstring and in BASELINE.md); report.py: tables and speedups.
- run_suite4.sh: full suite over the 4 GPUs of the gpupool node; run_baseline.sh: single-GPU baseline runs.
- BASELINE.md: FA4 / cuDNN / FA4-fp8 baselines (the targets); RESULTS.md: candidate results.
- ENV.md, install.sh, activate.sh: the shared env (torch cu130, CUDA 13.0.2 + ncu, cutlass-dsl, FA4 from source).
