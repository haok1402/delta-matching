#!/bin/bash
# Same-lease A/B on the evolve GPUs (fa4-bench).  usage: run_ab.sh SHA_A SHA_B
#   8k / 32k / 64k : one gpupool run per N with A then B back to back (same lease, same GPU) -> commit delta is same-lease.
#   128k           : one gpupool run per (commit, head layout) (lead / gpu-keeper 19:32: a 2-commit 128k chunk does not fit
#                    20 min).  The 128k A-vs-B pair is therefore CROSS-LEASE; delta reports must say so.
# Every run appends its records to disk as each shape finishes (--resume skips what is already there), so a cut run loses at
# most the shape in flight.  Timeouts from measured part runtimes: 8k/32k chunk (2 commits) 2-4 min -> 8m; 64k chunk 4-6 min
# -> 12m; one 128k part (1 commit x 1 head) 107-176 s warm, up to ~5 min with a cold JIT compile -> 8m.
# env: ND=1 adds the non-deterministic arm; GATE=1 adds --gate; ENV_B="K=V ..." extra env for commit B only; NSET="N ..."
#      restricts lengths; ONLY="sha:H/Hk ..." restricts the 128k parts; PFX result prefix (default focus); R2=1 re-times A after B (noise floor).
# results: results/${PFX:-focus}_<A>_<B>_<sha>[_r2].s0.jsonl  (same names as before -> delta.py / headline.py unchanged)
A=$1; B=$2
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
C=$PWD/cands/kda_full.py:attn
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/kda/env/dslcache}
NS=${NSET:-$(python3 -c "import json;print(' '.join(str(n) for n in sorted({x['N'] for x in json.load(open('headline_shapes.json'))['shapes']})))")}
HEADS=$(python3 -c "import json;print(' '.join(sorted({'%d/%d' % (x['H'], x['Hk']) for x in json.load(open('headline_shapes.json'))['shapes']})))")
TAGS="${A} ${B}"; [ "${R2:-0}" = 1 ] && TAGS="$TAGS ${A}_r2"

part() {   # part <N> <tags> <heads or ""> <timeout> <extra attnbench args>
  local N=$1 tags=$2 heads=$3 tmo=$4 xa=$5
  GPUPOOL_WHO=fa4-bench ../gpupool/bin/gpupool run --gpus 1 ${GATE:+--gate} --timeout $tmo --retry 2 --shell -- "
source ../env/activate.sh
for tag in $tags; do
  sha=\${tag%_r2}
  envb=; [ \"\$sha\" = \"$B\" ] && envb=\"$ENV_B\"
  env \$envb KDA_SNAP=\$PWD/snap/\$sha python attnbench.py --arms fa4,fa4_det --cand dm_matched=$C@matched ${ND:+--cand dm_nd=$PWD/cands/kda_full_nd.py:attn@matched} --suite focus --seqlens $N ${heads:+--heads $heads} $xa --tag ab-\$tag --out results/${PFX:-focus}_${A}_${B}_\$tag.s0.jsonl --resume >> results/${PFX:-focus}_${A}_${B}_\$tag.s0.log 2>&1 || exit \$?
done"
  local rc=$?; [ $rc = 0 ] || echo "part N=$N tags=$tags heads=$heads rc=$rc"
}

for N in $NS; do
  if [ "$N" -ge 131072 ]; then
    for tag in $TAGS; do
      for hk in $HEADS; do
        [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $tag:$hk " && continue   # ONLY="sha:H/Hk ...": run just these parts
        t128=${TMO128:-8m}; [ "$hk" = 32/4 ] && t128=${TMO128_32:-12m}   # 32/4 part measured >480 s (46a5)
        part $N "$tag" "$hk" $t128 "--cupti-iters 0 --min-iters 10 --warmup 2 --sustain-s 0.5"
      done
    done
  else
    tmo=${TMO:-8m}; [ "$N" -ge 65536 ] && tmo=12m
    part $N "$TAGS" "" $tmo ""
  fi
done
