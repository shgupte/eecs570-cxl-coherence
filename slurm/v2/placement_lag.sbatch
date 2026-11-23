#!/usr/bin/env bash
#SBATCH --job-name=pl-lag
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=01:30:00
#SBATCH --output=logs/pl-lag-%j.log
# Placement with a lagged sole-updater table (pipelined placer), its effect
# on the step (link model), and the placer's speed with 1 and 16 copies.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3; A=analysis
D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
PL="--batches $D/batches/d1 --policy affinity --cmax 512 --beta 2 --passes 0 --fast --steps 32"
$py $A/sample_placement.py $PL --out $D/cache/pl_lag1 --lag 1 | tail -n 1
for l in 4 8 16; do $py $A/sample_placement.py $PL --out $D/cache/pl_lag$l --lag $l | tail -n 1 & done; wait
echo "## 16 placers at once (speed under contention)"
for j in $(seq 16); do $py $A/sample_placement.py $PL --out $TMPDIR/pl$j --lag 1 | tail -n 1 & done; wait
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
for c in d1 pl_lag1 pl_lag4 pl_lag8 pl_lag16; do
  ( echo "## $c"; $py $A/linkmodel.py --sample 1 --cache $D/cache/$c --first 18 --steps 14 --skip 4 \
      --designs shard,shard_mig,comm_keep,mesi_comm --gpu-bw 63 --pool-bw 504 --nv-bw 63 $CXL ) &
done; wait
