#!/usr/bin/env bash
#SBATCH --job-name=linkmodel
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00
#SBATCH --output=logs/linkmodel-%j.log
# Full-batch link model with physical link figures (see analysis/linkmodel.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3
C=/scratch/engin_root/engin1/bany/dlrm_work/criteo
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
m() { echo "## $1"; shift; $py analysis/linkmodel.py --sample 1 "$@"; }
SH=shard,shard_mig
for nv in 25 50 300 450; do
  m "sharded d1 nv=$nv" --cache $C/cache/d1 --batches $C/batches/d1 --designs shard_sq,shard --nv-bw $nv &
  m "sharded pl_aff2 nv=$nv" --cache $C/cache/pl_aff2 --designs $SH --nv-bw $nv &
done
wait
for bw in 63 121; do for p in 1 2 4 8; do
  m "pool d1 gpu=$bw ports=$p" --cache $C/cache/d1 --designs unc_agg --gpu-bw $bw --pool-bw $((bw * p)) $CXL &
  m "pool pl_aff2 gpu=$bw ports=$p" --cache $C/cache/pl_aff2 --designs unc_agg,comm_keep,mesi_comm --gpu-bw $bw --pool-bw $((bw * p)) $CXL &
done; wait; done
for cr in 20000 50000 100000 200000 400000; do
  m "mesi cache_rows=$cr gpu=63 ports=8" --cache $C/cache/pl_aff2 --designs mesi_comm --gpu-bw 63 --pool-bw 504 $CXL --cache-rows $cr --steps 28 --skip 12 &
done
m "mesi unlimited 28 steps" --cache $C/cache/pl_aff2 --designs mesi_comm,comm_keep --gpu-bw 63 --pool-bw 504 $CXL --steps 28 --skip 12 &
wait
