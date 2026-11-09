#!/bin/bash
# Promotion validation of one candidate, split across the GPUs of the current gpupool reservation:
#   gpupool run --gpus N --who <you> -- bash validator/validate_all.sh /abs/cand.py <outdir>
# GPU 0: quick, then the sanitizer tier, then full, then quick with an empty compile cache (cold); the remaining GPUs: bench shards (with 1 GPU, everything in sequence).
# Exit 0 iff every part PASSes.  Each part writes <outdir>/<part>.{log,json}; summary in <outdir>/SUMMARY.txt.
set -u
CAND=$1; OUT=$2; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#G[@]}
run() { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python "$@" > "$OUT/$name.log" 2>&1; echo "$name rc=$?" >> "$OUT/rc.txt"; }
rm -f "$OUT/rc.txt"
(run ${G[0]} quick $HERE/validate.py "$CAND" --shapes quick --json "$OUT/quick.json"
 run ${G[0]} sanitize $HERE/sanitize.py "$CAND" --log "$OUT/sanitize_memcheck.txt"
 run ${G[0]} racecheck $HERE/sanitize.py "$CAND" --tool racecheck --log "$OUT/sanitize_racecheck.txt"
 run ${G[0]} full $HERE/validate.py "$CAND" --shapes full --json "$OUT/full.json"
 # cold compile cache (CuTe DSL + Triton): abf6f87's host fast path recorded wrong replay plans on calls that JIT-compiled (warm caches hid it)
 CC="$OUT/coldcache.tmp"; rm -rf "$CC"
 FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$CC" TRITON_CACHE_DIR="$CC/triton" run ${G[0]} cold $HERE/validate.py "$CAND" --shapes quick --json "$OUT/cold.json"
 rm -rf "$CC"
 if [ "$NG" -eq 1 ]; then run ${G[0]} bench $HERE/validate.py "$CAND" --shapes bench --json "$OUT/bench.json"; fi) &
if [ "$NG" -gt 1 ]; then
  NS=$((NG - 1))
  for i in $(seq 0 $((NS - 1))); do
    run ${G[$((i + 1))]} bench_$i $HERE/validate.py "$CAND" --shapes bench --shard $i/$NS --json "$OUT/bench_$i.json" &
  done
fi
wait
{ for f in "$OUT"/*.log; do echo "$(basename "$f" .log): $(grep -hE '^(PASS|FAIL)' "$f" | tail -1)"; done; } > "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"
if grep -q -v "rc=0" "$OUT/rc.txt"; then echo "OVERALL: FAIL"; exit 1; fi
echo "OVERALL: PASS"
