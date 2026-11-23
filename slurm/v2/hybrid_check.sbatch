#!/usr/bin/env bash
#SBATCH --job-name=hybrid
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=00:40:00
#SBATCH --output=logs/hybrid-%j.log
# Hybrid (GPUs keep whole bags, the pool pools the rest) against the pure
# near-memory pool. Multi-hot bags over the one-hot caches (see hybrid_flows).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib:analysis
py=/home/bany/bin/python3; D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
DES=hyb_nopromo,hyb,hyb_nopromo_dedup,hyb_dedup
for c in pl_aff2 d1 pl_lag8; do for p in 8 2; do
  ( echo "## $c ports=$p"; $py analysis/linkmodel.py --sample 1 --cache $D/cache/$c --designs $DES --gpu-bw 63 --pool-bw $((63 * p)) $CXL --steps 14 --skip 5 ) > logs/hybrid-$c-$p.txt 2>&1 &
done; done; wait
cat logs/hybrid-*-*.txt
