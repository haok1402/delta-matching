#!/bin/bash
# usage: build.sh <tag>...   DM_DBG=1 + v5 build of .local/wt/v5-<tag> into .local/jit/v5-<tag>-dbg
R=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching; L=$R/.local
for t in "$@"; do
  ( cd $L/wt/v5-$t && DM_DBG=1 DM_FA_SRC=$R/third_party/flash-attention MAX_JOBS=32 DM_JIT_DIR=$L/jit/v5-$t-dbg \
      python -m delta_matching.jit > $L/logs/v5-build-$t-dbg.log 2>&1; echo "$t: $(tail -n 1 $L/logs/v5-build-$t-dbg.log | cut -c1-60)" )
done
