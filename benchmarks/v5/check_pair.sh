#!/bin/bash
# usage: check_pair.sh <orig_tag> <cand_tag>  -> ~/workspace/haok1402/kda-results/r21-v5/<cand>_vs_<orig>/*.json
S=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/rounds/r21-v5
o=$1; c=$2; O=~/workspace/haok1402/kda-results/r21-v5/${c}_vs_${o}; mkdir -p $O
for f in cases_default cases_kv cases_head cases_streams2 accept stress; do
  python $S/numerics_v5.py check --orig $S/data/$o/$f.pt --cand $S/data/$c/$f.pt --worst 5 ${ENV:+--env $S/data/lsep/$f.pt $S/data/lsem/$f.pt} --report $O/$f.json > $O/$f.log 2>&1
  echo "$c vs $o $f: $(tail -n 1 $O/$f.log)"
done
