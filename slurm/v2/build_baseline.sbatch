#!/usr/bin/env bash
#SBATCH --job-name=cxl-base-build
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build-baseline-%j.log
set -euo pipefail
module load gcc/11.2.0 python/3.11.5
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "${SIMCXL:-/scratch/engin_root/engin1/bany/simcxl-baseline}"
scons defconfig build/X86_CXL_MESI build_opts/X86
scons setconfig build/X86_CXL_MESI RUBY_PROTOCOL_CXL_MESI_TWO_LEVEL=y
scons build/X86_CXL_MESI/gem5.opt -j8
