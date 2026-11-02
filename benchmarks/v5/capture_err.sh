#!/bin/bash
# usage: capture_err.sh <tag>...  production-path numerics.py captures (errors vs fp32) on 4 routes -> data/<tag>/err_<route>.json
R=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching; L=$R/.local; S=$L/rounds/r21-v5
for t in "$@"; do
  case $t in base) root=$L/frozen/base;; r19_fwd) root=$L/frozen/r19_fwd;; *) root=$L/wt/v5-$t;; esac
  mkdir -p $S/data/$t
  for r in default kv head streams2; do
    PYTHONPATH=$root python $R/benchmarks/numerics.py capture --root $root --route $r --out $S/data/$t/err_$r.json > $S/data/$t/err_$r.log 2>&1
    echo "$t $r rc $? $(tail -n 1 $S/data/$t/err_$r.log | cut -c1-60)"
  done
done
