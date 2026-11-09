#!/bin/bash
# 128k headline part as 1-GPU leases (one per head config), both modes; results appended to hl_<sha>.g1<k>.jsonl
SHA=$1
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
C=$PWD/cands/kda_full.py:attn
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/kda/env/dslcache
i=${I0:-0}
for HK in ${HEADS:-32/4 16/4}; do
  GPUPOOL_WHO=fa4-bench ../gpupool/bin/gpupool run --gpus 1 ${GATE:+--gate} --timeout $([ "$HK" = 32/4 ] && echo 12m || echo 8m) --retry 2 --shell -- "source ../env/activate.sh && KDA_SNAP=\$PWD/snap/$SHA python attnbench.py --arms fa4,fa4_det --cand dm_matched=$C@matched --cand dm_nd=$PWD/cands/kda_full_nd.py:attn@matched --suite focus --seqlens 131072 --heads $HK --tag hl128-$SHA --out results/hl_$SHA.g1$i.jsonl --resume >> results/hl_$SHA.g1$i.log 2>&1" &
  i=$((i+1))
done
wait
