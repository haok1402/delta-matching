#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=kda-fuse-ab
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools/logs/%x-%j.log
# usage: sbatch --wait ab.sh OUTTAG REPS "BITWISE_TAGS" ROOT_A ROOT_B [ROOT_C ...]
# backward-only wall + per-kernel times (bwd_timeline.py) on the four acceptance shapes, alternating root order per
# repetition, fresh process per (root, rep); default-route byte equivalence vs the frozen baseline dump for BITWISE_TAGS.
set -uo pipefail
source /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/env.sh
T=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools
W=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse
OUTTAG=$1; REPS=$2; BW=$3; shift 3; ROOTS=("$@")
O=$T/out/$OUTTAG; mkdir -p $O; cd $T
echo "host $(hostname) job $SLURM_JOB_ID roots ${ROOTS[*]}"
for b in $BW; do
  R=$T/roots/$b
  PYTHONPATH=$R python $W/benchmarks/equivalence.py dump --route default --expect-root $R --out $TMPDIR/eq_$b.pt > $O/eq_$b.log 2>&1 \
    && python $W/benchmarks/equivalence.py compare $L/equiv/base3_default.pt $TMPDIR/eq_$b.pt > $O/cmp_$b.log 2>&1
  echo "BITWISE $b rc $? $(tail -1 $O/cmp_$b.log)"; rm -f $TMPDIR/eq_$b.pt
done
n=${#ROOTS[@]}
for ((r=0; r<REPS; r++)); do
  for ((i=0; i<n; i++)); do
    if (( r % 2 )); then j=$((n-1-i)); else j=$i; fi
    R=${ROOTS[$j]}; [ -d $R ] || R=$T/roots/$R
    tag=$(basename $R)
    PYTHONPATH=$R DM_SO=$R/delta_matching/_C.so python bwd_timeline.py ${SHAPES:+--shapes $SHAPES} --iters ${ITERS:-20} --traces 2 --out $O/${tag}_$r.json > $O/${tag}_$r.log 2>&1 || echo "RUN FAILED $tag $r"
  done
done
python summ.py $O ${ROOTS[@]}
