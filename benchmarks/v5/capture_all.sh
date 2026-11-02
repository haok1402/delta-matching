#!/bin/bash
# usage: capture_all.sh <tag> [suites...]  -> .local/rounds/r21-v5/data/<tag>/{cases_<route>,accept,stress}.pt
R=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching; L=$R/.local
S=$L/rounds/r21-v5; t=$1; shift; suites=${*:-"cases accept stress"}
D=$S/data/$t; mkdir -p $D
for s in $suites; do
  if [ $s = cases ]; then
    for r in default kv head streams2; do
      PYTHONPATH=$L/wt/v5-$t python $S/numerics_v5.py capture --root $L/wt/v5-$t --suite cases --route $r --out $D/cases_$r.pt > $D/cases_$r.log 2>&1; echo "$t cases $r rc $? $(tail -n 1 $D/cases_$r.log | cut -c1-80)"
    done
  else
    PYTHONPATH=$L/wt/v5-$t python $S/numerics_v5.py capture --root $L/wt/v5-$t --suite $s --out $D/$s.pt > $D/$s.log 2>&1; echo "$t $s rc $? $(tail -n 1 $D/$s.log | cut -c1-80)"
  fi
done
