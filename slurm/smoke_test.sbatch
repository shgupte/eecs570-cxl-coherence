#!/usr/bin/env bash
#SBATCH --job-name=cxl-smoke
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke-%j.log
set -euo pipefail

source "${CXL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/env.sh"
cxl_load_build_modules
export TRACE="$CXL_PROJECT_ROOT/traces/two-accel/litmus-accel0.csv"
export TRACE2="$CXL_PROJECT_ROOT/traces/two-accel/litmus-accel1.csv"
export NAME=litmus-smoke
bash "$CXL_PROJECT_ROOT/slurm/run_cxl_trace.sbatch"
python3 "$CXL_PROJECT_ROOT/scripts/check_litmus.py" \
    "$RESULT_ROOT/${NAME}-${SLURM_JOB_ID}/stats.txt"
