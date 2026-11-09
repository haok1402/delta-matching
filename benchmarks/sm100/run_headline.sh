#!/bin/bash
# Headline measurement for one commit: EVERY focus shape on EVERY one of the 4 GPUs (no sharding). Each GPU times FA4,
# FA4-det and the candidate (matched + stale) interleaved in rounds; headline.py takes the per-shape MEDIAN across GPUs,
# so the absolute ratio does not depend on which GPU a shape lands on (FA4 varies up to 7% per 8k shape between GPUs).
# usage: run_headline.sh SHA   -> results/hl_<sha>.g<i>.jsonl ; then: python headline.py hl_<sha>
SHA=$1
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
C=$PWD/cands/kda_full.py:attn
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/kda/env/dslcache}
GPUPOOL_WHO=fa4-bench exec ../gpupool/bin/gpupool run --gpus ${NG:-4} --timeout 20m --retry 2 --shell -- "
source ../env/activate.sh
IFS=, read -ra G <<< \"\$CUDA_VISIBLE_DEVICES\"
for ((i=0; i<\${#G[@]}; i++)); do
  KDA_SNAP=\$PWD/snap/$SHA CUDA_VISIBLE_DEVICES=\${G[\$i]} python attnbench.py --arms fa4,fa4_det --cand dm_matched=$C@matched --cand dm_stale=$C@stale --suite focus --tag hl-$SHA-gpu\${G[\$i]} --out results/hl_$SHA.g\$i.jsonl --resume > results/hl_$SHA.g\$i.log 2>&1 &
done
wait"
