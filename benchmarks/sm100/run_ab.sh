#!/bin/bash
# Same-lease A/B: commits A then B (then A again), each sharded over the SAME GPUs (shard i always on GPU i), so GPU
# placement and node are identical for both.  usage: run_ab.sh SHA_A SHA_B   (results/ab_<A>_<B>_<sha>[_r2].s<i>.jsonl)
A=$1; B=$2
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
C=$PWD/cands/kda_full.py:attn
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/kda/env/dslcache}
# NG default = GPUs free right now (on filler), so the request starts at once and blocks nobody (min 1)
NG=${NG:-$(../gpupool/bin/gpupool status 2>/dev/null | grep -c "filler util")}; [ "$NG" -ge 1 ] 2>/dev/null || NG=1
GPUPOOL_WHO=fa4-bench exec ../gpupool/bin/gpupool run --gpus $NG --timeout 2h --retry 5 --shell -- "
source ../env/activate.sh
IFS=, read -ra G <<< \"\$CUDA_VISIBLE_DEVICES\"
for tag in ${A} ${B} ${A}_r2; do
  sha=\${tag%_r2}
  for ((i=0; i<\${#G[@]}; i++)); do
    KDA_SNAP=\$PWD/snap/\$sha CUDA_VISIBLE_DEVICES=\${G[\$i]} python attnbench.py --arms fa4,fa4_det --cand dm_matched=$C@matched --cand dm_stale=$C@stale --suite main --shard \$i/\${#G[@]} --tag ab-\$tag --out results/ab_${A}_${B}_\$tag.s\$i.jsonl --resume > results/ab_${A}_${B}_\$tag.s\$i.log 2>&1 &
  done
  wait
done"
