#!/usr/bin/env bash
#SBATCH --job-name=criteo-reuse
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/criteo-reuse-%j.log
set -euo pipefail
PY=/home/bany/bin/python3
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
A=~/eecs570-cxl-coherence/analysis
C=/scratch/engin_root/engin1/bany/criteo1tb
W=/scratch/engin_root/engin1/bany/dlrm_work/criteo
# within-day: 32 consecutive batches of day 1
$PY $A/criteo_to_batches.py --day-dir "$C/day=2015-02-15" --out $W/batches/d1 --batches 32
# across days: first batch of days 2, 8, 24
for d in 2015-02-16 2015-02-22 2015-03-10; do
  $PY $A/criteo_to_batches.py --day-dir "$C/day=$d" --out $W/batches/$d --batches 1
done
mkdir -p $W/batches/drift
ln -sf $W/batches/d1/b0.npz $W/batches/drift/b0.npz
i=1; for d in 2015-02-16 2015-02-22 2015-03-10; do ln -sf $W/batches/$d/b0.npz $W/batches/drift/b$i.npz; i=$((i+1)); done
$PY $A/reuse_analysis.py --data-dir $W/batches/d1 --pattern 'b{i}.npz' --batches 32 --row-shift 26 \
  --cache-dir $W/cache/d1 --out $W/reuse_d1
$PY $A/reuse_analysis.py --data-dir $W/batches/drift --pattern 'b{i}.npz' --batches 4 --row-shift 26 \
  --cache-dir $W/cache/drift --out $W/reuse_drift
