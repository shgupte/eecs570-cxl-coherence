#!/usr/bin/env bash
#SBATCH --job-name=cxl-base-rebuild
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rebuild-baseline-%j.log
set -euo pipefail
module load gcc/11.2.0 python/3.11.5
source "${VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate"
cd "${SIMCXL:-/scratch/engin_root/engin1/bany/simcxl-baseline}"
scons build/X86_CXL_MESI/gem5.opt -j${SLURM_CPUS_PER_TASK:-8}
