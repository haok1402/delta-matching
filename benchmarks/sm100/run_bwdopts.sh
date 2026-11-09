#!/bin/bash
# babaudit re-measure: 94ec380 default vs 3 DM_BWD_OPTS knobs, interleaved in one process per N (det mode)
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/bench
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100/kda/env/dslcache
C=$PWD/cands
for N in 8192 32768 65536; do
  GPUPOOL_WHO=fa4-bench ../gpupool/bin/gpupool run --gpus 1 --timeout 20m --retry 2 --shell -- "source ../env/activate.sh && KDA_SNAP=\$PWD/snap/94ec380 python attnbench.py --arms fa4 --cand opt_default=$C/kda_bwdopts_default.py:attn@matched --cand opt_mskip=$C/kda_bwdopts_mskip.py:attn@matched --cand opt_nodqrd=$C/kda_bwdopts_nodqrd.py:attn@matched --cand opt_fullbar=$C/kda_bwdopts_fullbar.py:attn@matched --suite focus --seqlens $N --passes bwd,fwd_bwd --tag bwdopts --out results/bwdopts_94ec380.s0.jsonl --resume >> results/bwdopts_94ec380.s0.log 2>&1"
done
