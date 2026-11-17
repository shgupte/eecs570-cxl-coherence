#!/usr/bin/env bash
#SBATCH --job-name=mh-cache
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G --time=01:30:00
#SBATCH --output=logs/mh-cache-%j.log
# Multi-hot: MESI designs with a limited GPU cache (rows per GPU), 28 steps.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3; D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
for cr in 500000 1000000 2000000 4000000 0; do
  ( echo "## mh_aff cache_rows=$cr"; $py analysis/linkmodel.py --sample 1 --cache $D/cache/mh_aff --designs mesi_pm,mesi_comm --gpu-bw 63 --pool-bw 504 $CXL --cache-rows $cr --steps 28 --skip 12 ) > logs/mh-cache-$cr.txt 2>&1 &
done
( echo "## mh_aff software, 28 steps"; $py analysis/linkmodel.py --sample 1 --cache $D/cache/mh_aff --designs comm_keep,shard,shard_mig --gpu-bw 63 --pool-bw 504 --nv-bw 300 $CXL --steps 28 --skip 12 ) > logs/mh-cache-sw.txt 2>&1 &
wait; cat logs/mh-cache-*.txt
