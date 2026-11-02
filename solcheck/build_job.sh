#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_short,cpu,cpu_interactive
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/solcheck-runs/build-%j.log
# build_job.sh <outdir> <mix.json> [adjust.json]: synthetic + micro configs, parallel nvcc, SASS checks
set -e
source /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/.local/env.sh
S=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/kda-work/solcheck/solcheck
O=$1; mkdir -p $O; cd $O
python $S/cfgs.py --mix $2 --out cfgs.json ${3:+--adjust $3} > /dev/null
python $S/micro.py --cfgs cfgs.json --out micro.json
python - <<'PY'
import json
json.dump(json.load(open("cfgs.json")) + json.load(open("micro.json")), open("all.json", "w"), indent=1)
PY
python $S/gen.py --cfg all.json --out src > names
mkdir -p bin log sass; cat names | xargs -P ${NPAR:-8} -I{} bash -c 'nvcc -gencode arch=compute_90a,code=sm_90a -O3 -lineinfo -diag-suppress 177,550 -Xptxas -v -o bin/{} src/{}.cu > log/{}.ptxas 2>&1 || echo "FAIL {}"; cuobjdump -sass bin/{} > sass/{}.sass; cuobjdump -res-usage bin/{} | grep -o "REG:[0-9]* STACK:[0-9]*" > log/{}.res'
grep -l "C75" log/*.ptxas | sed 's/^/WARN /' || true
python $S/check.py --cfgs cfgs.json --dir sass
