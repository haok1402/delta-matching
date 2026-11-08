# source ~/workspace/haok1402/sm100/env/activate.sh
_E=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/env
export CUDA_HOME=$_E/cuda-13.0
export PATH=$CUDA_HOME/bin:$(ls -d $CUDA_HOME/nsight-compute-* 2>/dev/null | head -1):$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export UV_CACHE_DIR=$_E/.uv-cache UV_PYTHON_INSTALL_DIR=$_E/.uv-python
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-$_E/.cute-dsl-cache}
. $_E/venv/bin/activate
unset _E
# FA4 / CuTe-DSL persistent compile cache (avoids recompiling in every process; keeps reserved GPUs from idling on JIT)
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/env/.fa4-cache}
