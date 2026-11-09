#!/bin/bash
# List kda-sm100 commits newer than the last benchmarked snapshot, with dm-validator verdict lines and candidates.jsonl hits.
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching
git fetch -q origin kda-sm100-kernels 2>/dev/null
S=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/sm100
done_shas=$( (for f in $S/bench/results/full_*.s0.jsonl; do b=$(basename $f .s0.jsonl); echo ${b#full_}; done; ls $S/bench/results | sed -n -E 's/^(ab|focus)_[0-9a-f]+_([0-9a-f]+)_\2\.s0\.jsonl$/\2/p') | sort -u)   # benchmarked = has results
for c in $(git rev-list --abbrev-commit --max-count=15 origin/kda-sm100-kernels); do
  echo "$done_shas" | grep -q "^$c" && break
  v=$(grep -h "$c" $S/STATUS-dm-validator.md $S/inbox/fa4-bench.md 2>/dev/null | grep -iE "pass|valid" | head -2 | cut -c1-160)
  k=$(grep -c "$c" $S/kda/candidates.jsonl 2>/dev/null)
  echo "NEW $(git log --oneline -1 $c | cut -c1-120) | cand.jsonl:$k | validator: ${v:-none}"
done
