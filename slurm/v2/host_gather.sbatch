#!/usr/bin/env bash
#SBATCH --job-name=host-gather
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36 --mem=170G --time=00:40:00
#SBATCH --output=logs/host-gather-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.2.0
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib OMP_PROC_BIND=spread OMP_PLACES=cores
lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"
/home/bany/bin/python3 analysis/host_gather_bench.py --batches /scratch/engin_root/engin1/bany/dlrm_work/criteo/batches/d1 --threads 1,2,4,8,16,32,36
