#!/usr/bin/env bash
#SBATCH --job-name=cxl-shared-data
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shared-data-%j.log
set -euo pipefail

source "${CXL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/env.sh"
cxl_load_build_modules
python3 "$CXL_PROJECT_ROOT/scripts/prepare_shared_data.py" \
    --shared-root "$CXL_SHARED_ROOT" --dataset "$DATASET" "$@"
