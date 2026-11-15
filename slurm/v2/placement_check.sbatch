#!/usr/bin/env bash
#SBATCH --job-name=place-check
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=60G --time=02:00:00
#SBATCH --output=logs/place-check-%j.log
# DATA=<npz from staleness_train.py prep>; STEPS (40)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/analysis"
/home/bany/bin/python3 placement_check.py --data "$DATA" --steps "${STEPS:-40}" --threads 16
