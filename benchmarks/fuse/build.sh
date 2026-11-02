#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=kda-fuse-build
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools/logs/%x-%j.log
# usage: sbatch build.sh "TAG1=-DX=1 -DY=1" "TAG2=" ...   builds the fuse worktree per variant into roots/TAG
set -uo pipefail
source /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/env.sh
T=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse-tools
W=${WT:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/fuse}
export DM_FA_SRC=$L/../third_party/flash-attention MAX_JOBS=32 TORCH_CUDA_ARCH_LIST=9.0a
echo "host $(hostname) job $SLURM_JOB_ID worktree $W rev $(git -C $W rev-parse --short HEAD)"
for spec in "$@"; do
  TAG=${spec%%=*}; DEFS=${spec#*=}
  R=$T/roots/$TAG; rm -rf $R $T/jit/$TAG; mkdir -p $R
  ( cd $W && KDA_DEFS="$DEFS" DM_JIT_DIR=$T/jit/$TAG python -m delta_matching.jit ) > $T/logs/build_$TAG.log 2>&1
  if [ ! -f $T/jit/$TAG/_C.so ]; then echo "BUILD FAILED $TAG"; grep -m8 -i "error" $T/logs/build_$TAG.log; continue; fi
  rsync -a --exclude _C.so --exclude __pycache__ --exclude _jit $W/delta_matching $R/
  cp $T/jit/$TAG/_C.so $R/delta_matching/_C.so; cp $T/jit/$TAG/_build_info.py $R/delta_matching/ 2>/dev/null
  diag=$(grep -iE "spill|C7510|serializ" $T/logs/build_$TAG.log | grep -v " 0 bytes spill" | head -3)
  res=$(cuobjdump -res-usage $R/delta_matching/_C.so 2>/dev/null | grep -A1 -E "pass2_kernel_hd128|Fp8R|DeltaV6" | grep -o "REG:[0-9]* STACK:[0-9]* SHARED:[0-9]* LOCAL:[0-9]*" | sort | uniq -c | tr '\n' ';')
  echo "BUILT $TAG [$DEFS] sha $(sha256sum $R/delta_matching/_C.so | cut -c1-12) diag[$diag] res[$res]"
done
