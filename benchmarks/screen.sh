#!/bin/bash
# Quick screen of one candidate worktree (Hao, round 17: breadth first).  Runs INSIDE a 1-GPU sbatch job via run.sh:
#   sbatch --wait --time=0:45:00 .local/jobs/run.sh 'bash .local/jobs/screen.sh TAG WT [BITWISE] [SHAPES]'
# TAG: screen id (S17-xx); WT: worktree (relative to the repo); BITWISE=1: default-route byte equivalence vs the frozen
# baseline dump; SHAPES: "H:B:N,..." (default 16:2:8192,32:2:8192).  A = the accepted root (ACCEPTED, default r14_h3).
set -u
TAG=$1; WT=$2; BITWISE=${3:-1}; SHAPES=${4:-16:2:8192,32:2:8192}
R=$PWD; L=$R/.local; OUT=$L/screens/$TAG; ACC=${ACCEPTED:-$L/frozen/r14_h3}
rm -rf $OUT; mkdir -p $OUT
if [ ! -f $L/jit/$TAG/_C.so ]; then
  ( cd $WT && DM_FA_SRC=$R/third_party/flash-attention MAX_JOBS=${MAX_JOBS:-32} DM_JIT_DIR=$L/jit/$TAG python -m delta_matching.jit ) > $OUT/build.log 2>&1
fi
if [ ! -f $L/jit/$TAG/_C.so ]; then echo "SCREEN $TAG: BUILD FAILED"; grep -m5 -i "error" $OUT/build.log; exit 1; fi
touch $OUT/diag.txt; grep -iE "spill|C7510|serializ" $OUT/build.log | grep -v " 0 bytes spill" | head -3 > $OUT/diag.txt
ln -sfn $(realpath $L/jit/$TAG/_C.so) $WT/delta_matching/_C.so
[ -f $L/jit/$TAG/_build_info.py ] && cp $L/jit/$TAG/_build_info.py $WT/delta_matching/_build_info.py
cuobjdump -res-usage $L/jit/$TAG/_C.so 2>/dev/null | grep -A1 -E "Fq8F|Fp8R|DeltaV6|pass2_kernel_hd128" | grep -o "REG:[0-9]* STACK:[0-9]*" | sort | uniq -c > $OUT/res.txt
if [ "$BITWISE" = 1 ]; then
  PYTHONPATH=$WT python benchmarks/equivalence.py dump --route default --expect-root $WT --out $OUT/default.pt > $OUT/dump.log 2>&1 \
    && python benchmarks/equivalence.py compare $L/equiv/base3_default.pt $OUT/default.pt > $OUT/compare.log 2>&1
  echo "bitwise default route rc $? ($(tail -1 $OUT/compare.log))"
  rm -f $OUT/default.pt   # ~3.4 GB: the compare log is the record (lustre quota)
fi
for s in ${SHAPES//,/ }; do
  IFS=: read H B N <<< "$s"
  python benchmarks/ab.py run --a $ACC --b $WT --grid one --D 128 --B $B --H $H --Hk 4 --N $N --modes causal-matched \
    --reps 2 --smoke --out-dir $OUT/ab_H${H}_N${N} > $OUT/ab_H${H}_N${N}.log 2>&1
  python - $OUT/ab_H${H}_N${N} <<'PY'
import json, sys, statistics
d = sys.argv[1]
def med(p):
    rs = [json.load(open(f"{d}/{p}_{i}.json"))["records"][0] for i in range(2)]
    return {c: statistics.median(r["ms"][c] for r in rs) for c in rs[0]["ms"]}, statistics.median(r["e2e_fwdbwd_ms"] for r in rs)
(a, ea), (b, eb) = med("A"), med("B")
print(f"  {d.split('/')[-1]}: " + " ".join(f"{c} {100*(b[c]/a[c]-1):+.2f}%" for c in ("fwd", "pass0", "pass1", "pass2", "glue") if a[c] > 0)
      + f" | e2e fwd+bwd {100*(eb/ea-1):+.2f}%")
PY
done
echo "SCREEN $TAG done: diag [$(cat $OUT/diag.txt | head -1)] res $(cat $OUT/res.txt | tr '\n' ' ')"
