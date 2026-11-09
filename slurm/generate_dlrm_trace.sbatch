#!/usr/bin/env bash
#SBATCH --job-name=dlrm-trace
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dlrm-trace-%j.log

set -euo pipefail

source "${CXL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/env.sh"
cxl_load_trace_modules
: "${DATASET:?Set DATASET to the shared dataset path in env.local.sh}"
[[ -r "$DATASET" ]] || { echo "Dataset is not readable: $DATASET" >&2; exit 1; }
python3 "$CXL_PROJECT_ROOT/scripts/generate_dlrm_trace.py" "$DATASET" "$OUTPUT_DIR"
