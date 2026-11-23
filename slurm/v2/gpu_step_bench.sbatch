#!/usr/bin/env bash
#SBATCH --job-name=gpu-step
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=gpu_mig40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=160G --time=01:00:00
#SBATCH --output=logs/gpu-step-%j.log
# Measured DLRM step on a real GPU (analysis/gpu_step_bench.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
lscpu | grep "Model name"
/home/bany/bin/python3 analysis/gpu_step_bench.py --batches $D/batches/d1 --cache $D/cache/d1 ${ARGS:-}
