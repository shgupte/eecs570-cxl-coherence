#!/usr/bin/env bash
#SBATCH --job-name=pooled
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=01:00:00
#SBATCH --output=logs/pooled-%j.log
# Is row migration still ahead when the sharded baseline ships pooled vectors
# (what TorchRec / HugeCTR do for multi-hot features)? Also: memory balance.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib:analysis
py=/home/bany/bin/python3; D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
for c in d1 pl_aff2; do
  ( echo "## one-hot $c"; $py analysis/linkmodel.py --sample 1 --cache $D/cache/$c --batches $D/batches/d1 --designs pooled,pooled_dedup,shard,shard_mig --nv-bw 300 ) > logs/pooled-$c.txt 2>&1 &
done
for c in mh_d1 mh_aff; do
  ( echo "## multi-hot $c"; $py analysis/linkmodel.py --sample 1 --cache $D/cache/$c --batches $D/batches/d1 --bag-sizes --designs pooled,pooled_dedup,shard,shard_mig --nv-bw 300 ) > logs/pooled-$c.txt 2>&1 &
done
wait; cat logs/pooled-*.txt
