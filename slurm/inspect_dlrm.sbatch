#!/usr/bin/env bash
#SBATCH --job-name=dlrm-inspect
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dlrm-inspect-%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load python/3.10.4 pytorch/2.0.1

DATASET="${DATASET:-/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shgupte/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_0.pt.gz}"
OUTPUT="${OUTPUT:-/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shgupte/dlrm_work/inspection.json}"

python scripts/inspect_dlrm.py "$DATASET" "$OUTPUT"
