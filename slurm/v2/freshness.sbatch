#!/usr/bin/env bash
#SBATCH --job-name=fresh
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=gpu_mig40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=60G --time=01:00:00
#SBATCH --output=logs/fresh-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
/home/bany/bin/python3 analysis/freshness.py --data /scratch/engin_root/engin1/bany/dlrm_work/stale/d15_40.npz ${ARGS:-}
