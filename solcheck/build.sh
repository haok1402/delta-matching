#!/bin/bash
# build.sh <cfgs.json> <outdir>: generate, compile (sm_90a) and dump SASS of every synthetic kernel
set -e
D=$(cd "$(dirname "$0")" && pwd)
python $D/gen.py --cfg $1 --out $2 > $2.names 2>/dev/null || { mkdir -p $2; python $D/gen.py --cfg $1 --out $2 > $2.names; }
cd $2
for n in $(cat $2.names); do
  nvcc -gencode arch=compute_90a,code=sm_90a -O3 -lineinfo -diag-suppress 177,550 -o $n $n.cu 2> $n.ptxas || { cat $n.ptxas; exit 1; }
  cuobjdump -sass $n > $n.sass
  grep -E "C75|error" $n.ptxas | sed "s/^/$n: /" || true
done
