#!/usr/bin/env bash
#SBATCH --job-name=cxl-build
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build-%j.log
set -euo pipefail

source "${CXL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/env.sh"
bash "$CXL_PROJECT_ROOT/setup.sh"
