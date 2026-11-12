#!/usr/bin/env bash
#SBATCH --job-name=dlrm-reuse
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/reuse-%j.log
set -euo pipefail
module load python/3.10.4 pytorch/2.0.1
B=/scratch/engin_root/engin1/bany
python ~/eecs570-cxl-coherence/analysis/reuse_analysis.py --data-dir $B/dlrm_datasets/2021 \
  --cache-dir $B/dlrm_work/reuse_cache --out $B/dlrm_work/reuse
