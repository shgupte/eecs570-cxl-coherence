#!/usr/bin/env bash
#SBATCH --job-name=local-memory
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/local-memory-%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/11.2.0 python/3.11.5

GEM5="/scratch/eecs570f26s001_class_root/eecs570f26s001_class/nshreyas/cxl-coherence/sources/SimCXL/build/X86_CXL_MESI/gem5.opt"

"$GEM5" \
    -d "results/local-memory-${SLURM_JOB_ID}" \
    configs/local_memory.py
