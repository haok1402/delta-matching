#!/bin/bash
# Full suite over NG GPUs (default 4; NG=2 if only 2 are free), every GPU busy: one attnbench process per GPU (--shard i/4), each timing all arms interleaved
# on its own GPU (ratios per GPU). Resumable (--resume) and auto-relaunched on timeout/node loss (--retry).
# usage: run_suite4.sh TAG OUT "ATTNBENCH ARGS"   e.g. run_suite4.sh cand1 results/cand1.jsonl "--arms fa4 --cand c1=/abs/c.py:attn"
TAG=$1; OUT=$2; ARGS=$3
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
# NG default = GPUs free right now (on filler), so the request starts at once and blocks nobody (min 1)
NG=${NG:-$(../gpupool/bin/gpupool status 2>/dev/null | awk '/^\[/{act=($0 ~ /ACTIVE/)} act && /filler util/{n++} END{print n+0}')}  # ACTIVE node only
[ "$NG" -ge 1 ] 2>/dev/null || NG=1; [ "$NG" -le 4 ] || NG=4
GPUPOOL_WHO=fa4-bench exec ../gpupool/bin/gpupool run --gpus $NG --warmup ${WARMUP:-5m} --timeout 2h --retry 5 --shell -- "
source ../env/activate.sh
IFS=, read -ra G <<< \"\$CUDA_VISIBLE_DEVICES\"
for ((i=0; i<\${#G[@]}; i++)); do
  CUDA_VISIBLE_DEVICES=\${G[\$i]} python attnbench.py $ARGS --suite \${SUITE:-focus} --shard \$i/\${#G[@]} --tag $TAG --out ${OUT%.jsonl}.s\$i.jsonl --resume > ${OUT%.jsonl}.s\$i.log 2>&1 &
done
wait"
