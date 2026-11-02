#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=kda-fuse-t1
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools/logs/%x-%j.log
set -uo pipefail
source /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/env.sh
T=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools
ACC=$L/frozen/r17_p1
cd $T
echo "host $(hostname) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=name,clocks.sm --format=csv
export PYTHONPATH=$ACC DM_SO=$ACC/delta_matching/_C.so
S8=16:2:8192,32:2:8192
for rep in 1 2; do
python bwd_timeline.py --shapes $S8 --iters 40 --traces 1 --out out/t1_def_$rep.json
for g in head kv; do for s in 1 2 4; do
DM_BWD_HEAD_GROUPS=$g DM_BWD_STREAMS=$s python bwd_timeline.py --shapes $S8 --iters 40 --traces 1 --out out/t1_${g}_s${s}_$rep.json
done; done
done
