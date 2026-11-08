#!/bin/bash
# Full suite over 4 GPUs, every GPU busy: one attnbench process per GPU (--shard i/4), each timing all arms interleaved
# on its own GPU (ratios per GPU). Resumable (--resume) and auto-relaunched on timeout/node loss (--retry).
# usage: run_suite4.sh TAG OUT "ATTNBENCH ARGS"   e.g. run_suite4.sh cand1 results/cand1.jsonl "--arms fa4 --cand c1=/abs/c.py:attn"
TAG=$1; OUT=$2; ARGS=$3
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
GPUPOOL_WHO=fa4-bench exec ../gpupool/bin/gpupool run --gpus 4 --warmup ${WARMUP:-5m} --timeout 2h --retry 5 --shell -- "
source ../env/activate.sh
IFS=, read -ra G <<< \"\$CUDA_VISIBLE_DEVICES\"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=\${G[\$i]} python attnbench.py $ARGS --suite \${SUITE:-main} --shard \$i/4 --tag $TAG --out ${OUT%.jsonl}.s\$i.jsonl --resume > ${OUT%.jsonl}.s\$i.log 2>&1 &
done
wait"
