#!/bin/bash
# usage: run_baseline.sh TAG OUT [extra gpupool flags]   — resumable (--resume), relaunched by --retry on timeout/node loss
TAG=$1; OUT=$2; shift 2
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
GPUPOOL_WHO=fa4-bench exec ../gpupool/bin/gpupool run --gpus 1 --warmup 5m --timeout 2h --retry 5 "$@" --shell -- \
  "source ../env/activate.sh && python attnbench.py --arms ${ARMS:-fa4,cudnn,fa4_fp8} --suite main --tag $TAG --out $OUT --resume"
