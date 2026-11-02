#!/bin/bash
# usage: check_abs.sh <orig_tag> <cand_tag>  -> ~/workspace/haok1402/kda-results/r21-v5/<cand>_vs_<orig>/abs_*.json
S=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/rounds/r21-v5
o=$1; c=$2; O=~/workspace/haok1402/kda-results/r21-v5/${c}_vs_${o}; mkdir -p $O
for f in cases_default cases_kv cases_head cases_streams2 accept stress; do
  python $S/numerics_v5.py check --form v5-abs --orig $S/data/$o/$f.pt --cand $S/data/$c/$f.pt --worst 5 --report $O/abs_$f.json > $O/abs_$f.log 2>&1
  echo "$c vs $o $f: $(tail -n 1 $O/abs_$f.log)"
done
