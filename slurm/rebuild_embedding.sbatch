#!/usr/bin/env bash
#SBATCH --job-name=embedding-build
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embedding-build-%j.log

set -euo pipefail

source "$SLURM_SUBMIT_DIR/env.sh"
cxl_load_build_modules
source "$CXL_VENV/bin/activate"

cd "$SIMCXL_DIR"


scons build/X86_EMBEDDING/gem5.opt \
    EXTRAS="$CXL_PROJECT_ROOT/simobjects/embedding" \
    -j "$SLURM_CPUS_PER_TASK"