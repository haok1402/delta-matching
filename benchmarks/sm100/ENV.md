# sm100 shared env (owner: fa4-bench)

    source ~/workspace/haok1402/sm100/env/activate.sh

- Python 3.12 venv `venv/`: torch 2.14.1+cu130 (cuDNN 9.24), triton 3.8.0, nvidia-cutlass-dsl 4.8.1 (cu13 libs),
  FA4 = flash-attn-4 editable from `src/flash-attention` (Dao-AILab main @ b2edfab9), quack-kernels, numpy, pytest, nvidia-ml-py.
  Full list: `freeze.txt`.
- CUDA 13.0.2 toolkit `cuda-13.0/` (nvcc, ptxas, cuobjdump, Nsight Compute `ncu`), on PATH after activate.
  Matches the node driver branch (580). CUDA_HOME is set.
- Rebuild / repair: `sbatch install.sh` (CPU job, idempotent, resumable). Do not pip install into this venv ad hoc;
  ask fa4-bench (inbox/fa4-bench.md), or make your own venv with `uv venv --system-site-packages`-style overlays.
- GPU work through `sm100/gpupool/bin/gpupool run` (see gpupool/README.md), e.g.
  `gpupool run --gpus 1 -- bash -c 'source .../env/activate.sh && python x.py'` (or pass the venv python directly).
