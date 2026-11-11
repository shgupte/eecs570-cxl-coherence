#!/usr/bin/env bash
#SBATCH --job-name=sharded-memory
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sharded-memory-%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/11.2.0 python/3.11.5

GEM5="/scratch/eecs570f26s001_class_root/eecs570f26s001_class/nshreyas/cxl-coherence/sources/SimCXL/build/X86_CXL_MESI/gem5.opt"

RESULT_DIR="results/sharded-memory-${SLURM_JOB_ID}"

"$GEM5" \
    -d "$RESULT_DIR" \
    configs/sharded_memory.py

python3 scripts/summarize_sharded_memory.py "$RESULT_DIR" \
    | tee "$RESULT_DIR/summary.txt"
