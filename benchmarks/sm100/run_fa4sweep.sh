#!/bin/bash
# FA4 backward-config sweep on the target shapes (is FA4's default its fastest non-deterministic backward?)
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
for N in 8192 32768 65536; do
  GPUPOOL_WHO=fa4-bench ../gpupool/bin/gpupool run --gpus 1 --timeout 20m --retry 2 --shell -- "source ../env/activate.sh && python attnbench.py --arms fa4,fa4_det,fa4v:tn64,fa4v:pg1,fa4v:pg0,fa4v:no2cta --suite focus --seqlens $N --passes bwd,fwd_bwd --tag fa4sweep --out results/fa4sweep.s0.jsonl --resume >> results/fa4sweep.s0.log 2>&1"
done
