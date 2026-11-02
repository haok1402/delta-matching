#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=kda-fuse-t0
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools/logs/%x-%j.log
set -uo pipefail
source /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/env.sh
T=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools
ACC=$L/frozen/r17_p1
cd $T
echo "host $(hostname) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=name,clocks.sm --format=csv
export PYTHONPATH=$ACC DM_SO=$ACC/delta_matching/_C.so
python bwd_timeline.py --kernels --out out/t0_default.json
DM_BWD_STREAMS=2 python bwd_timeline.py --shapes 16:1:65536,32:1:65536 --out out/t0_streams2.json
DM_BWD_STREAMS=3 python bwd_timeline.py --shapes 16:1:65536,32:1:65536 --out out/t0_streams3.json
DM_BWD_WS_LIMIT_GB=16 python bwd_timeline.py --shapes 16:1:65536,32:1:65536 --out out/t0_kv.json
DM_BWD_WS_LIMIT_GB=16 DM_BWD_STREAMS=2 python bwd_timeline.py --shapes 16:1:65536,32:1:65536 --out out/t0_kv_s2.json
DM_DEBUG=1 python bwd_timeline.py --shapes 16:2:8192,16:1:65536 --iters 4 --traces 1 --out out/t0_dbg.json 2>&1 | grep -m 12 "delta_matching\|smem\|grid" 
