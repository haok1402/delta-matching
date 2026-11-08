#!/bin/bash
# Shared sm100 env: CUDA 13.0 toolkit (nvcc + Nsight Compute) + uv venv (torch cu130, triton, cutlass-dsl, FA4).
#SBATCH -A nvr_elm_llm -p cpu -q cpu-short -t 1:30:00 -c 8 --mem=32G -J fa4b-env -o /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/env/install.%j.log
set -euxo pipefail
E=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/env
cd $E
export UV_CACHE_DIR=$E/.uv-cache UV_PYTHON_INSTALL_DIR=$E/.uv-python TMPDIR=$E/tmp
mkdir -p $TMPDIR dl
# 1) CUDA 13.0.2 toolkit (sbsa = aarch64 server), toolkit only, includes nsight-compute
RUN=dl/cuda_13.0.2_580.95.05_linux_sbsa.run
# resumable: partial download lives in $RUN.part (curl -C - continues it); final name only after a complete download
if [ ! -s $RUN ] || [ "$(stat -c %s $RUN)" != 4071205419 ]; then
  [ -s $RUN ] && [ ! -e $RUN.part ] && mv $RUN $RUN.part
  curl -sSL -C - -o $RUN.part https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux_sbsa.run
  mv $RUN.part $RUN
fi
if [ ! -x cuda-13.0/bin/nvcc ]; then
  sh $RUN --silent --toolkit --toolkitpath=$E/cuda-13.0 --no-opengl-libs --no-man-page --no-drm --tmpdir=$TMPDIR
fi
ls cuda-13.0/bin | head; ls -d cuda-13.0/nsight-compute* || true
# 2) python env
[ -d venv ] || uv venv --python 3.12 venv
. venv/bin/activate
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
uv pip install numpy einops ninja pytest pytest-xdist pandas packaging setuptools wheel nvidia-ml-py
uv pip install -e "src/flash-attention/flash_attn/cute[cu13]"
python - <<'PY'
import torch, triton, cutlass
print("torch", torch.__version__, torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("triton", triton.__version__, "cutlass-dsl", getattr(cutlass, "__version__", "?"))
import flash_attn.cute as fc; print("fa4 ok", fc.__file__)
PY
uv pip freeze > $E/freeze.txt
