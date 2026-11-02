#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/solcheck-runs/run-%j.log
# run_job.sh <builddir> <outdir> [names-file] [launches] [reps] [realclock 0|1]
set -e
source /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/env.sh
W=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/solcheck
B=$1; O=$2; NAMES=${3:-$B/names}; L=${4:-20}; R=${5:-5}; RC=${6:-1}
mkdir -p $O; cd $O
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 50 > smi.csv &
SMI=$!
trap "kill $SMI 2>/dev/null || true" EXIT
python $W/benchmarks/calibrate.py --out calib.json
nvidia-smi -q -d CLOCK,POWER > smi_q.txt || true
FROZEN=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/${REAL_ROOT:-r17_p0s}
if [ "${REAL:-0}" = 1 ]; then   # the real kernels in the same allocation, same calibration
  echo "{\"mark\": \"real\", \"t\": $(date +%s.%N)}" > real_marks.jsonl
  PYTHONPATH=$FROZEN:$PYTHONPATH python $W/benchmarks/kernels.py --grid paper --modes causal-matched --calib calib.json \
    --out real.json 2>&1 | grep -v -iE "warn" | tail -3
  echo "{\"mark\": \"end\", \"t\": $(date +%s.%N)}" >> real_marks.jsonl
  python $W/benchmarks/sol4.py model --runs real.json --calib calib.json --counts $W/results/r17/accepted_p0s/sol4v2_counts.json \
    --out real_sol4.json --md real_sol4.md --grid paper | head -14
fi
for n in $(cat $NAMES); do
  echo "{\"mark\": \"$n\", \"t\": $(date +%s.%N)}" >> runs.jsonl
  timeout 300 $B/bin/$n 0 $L $R >> runs.jsonl || echo "{\"kernel\": \"$n\", \"error\": \"exit $?\"}" >> runs.jsonl
done
echo "{\"mark\": \"end\", \"t\": $(date +%s.%N)}" >> runs.jsonl
if [ "$RC" = 1 ]; then
  for n in ${SUSTAIN:-syn_fwd_dep syn_pass1_dep syn_pass0_dep syn_fwd_dep_z syn_pass1_dep_z t_SS128_w2 t_SS128_w2_z}; do   # sustained: ~3 s each
    echo "{\"mark\": \"sustain_$n\", \"t\": $(date +%s.%N)}" >> sustain.jsonl
    $B/bin/$n 0 150 10 | sed 's/"kernel"/"sustain": 1, "kernel"/' >> sustain.jsonl
  done
  echo "{\"mark\": \"end\", \"t\": $(date +%s.%N)}" >> sustain.jsonl
  PYTHONPATH=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/frozen/r17_p0s:$PYTHONPATH \
    python $W/solcheck/realclock.py --secs 6 > realclock.jsonl
fi
kill $SMI || true
echo done
