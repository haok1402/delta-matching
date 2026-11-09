#!/bin/bash
# Same-lease A/B on the single evolve GPU, chunked (gpu-keeper: evolve runs <= 20 min): ONE gpupool run per sequence
# length, each timing A then B on the same GPU back to back (both against interleaved FA4 / FA4-det), --resume + --retry 2,
# so other agents' requests interleave between chunks.  usage: run_ab.sh SHA_A SHA_B  [R2=1: A again after B]
# ENV_B="K=V ...": extra env for commit B only (e.g. ENV_B="DM_FWD_OPTS=ss_pv=1,kvs=6,lean=1").
# results: results/${PFX:-focus}_<A>_<B>_<sha>[_r2].s0.jsonl  (same names as before -> delta.py / headline.py unchanged)
A=$1; B=$2
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
C=$PWD/cands/kda_full.py:attn
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/kda/env/dslcache}
NS=$(python3 -c "import json;print(' '.join(str(n) for n in sorted({x['N'] for x in json.load(open('headline_shapes.json'))['shapes']})))")
TAGS="${A} ${B}"; [ "${R2:-0}" = 1 ] && TAGS="$TAGS ${A}_r2"
for N in $NS; do
  GPUPOOL_WHO=fa4-bench ../gpupool/bin/gpupool run --gpus 1 --timeout 20m --retry 2 --shell -- "
source ../env/activate.sh
for tag in $TAGS; do
  sha=\${tag%_r2}
  envb=; [ \"\$sha\" = \"$B\" ] && envb=\"$ENV_B\"
  env \$envb KDA_SNAP=\$PWD/snap/\$sha python attnbench.py --arms fa4,fa4_det --cand dm_matched=$C@matched --cand dm_stale=$C@stale ${ND:+--cand dm_nd=$PWD/cands/kda_full_nd.py:attn@matched} --suite focus --seqlens $N --tag ab-\$tag --out results/${PFX:-focus}_${A}_${B}_\$tag.s0.jsonl --resume >> results/${PFX:-focus}_${A}_${B}_\$tag.s0.log 2>&1 || exit \$?
done"
  rc=$?; [ $rc = 0 ] || echo "chunk N=$N rc=$rc"
done
