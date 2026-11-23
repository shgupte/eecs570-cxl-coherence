#!/usr/bin/env bash
#SBATCH --job-name=gen-x8
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=60G --time=02:00:00
#SBATCH --output=logs/gen-x8-%j.log
# 1/8-sample traces for the scale check of run_port_matrix.sh (X=8).
# CACHE=<dir with d1, pl_aff2>  BATCHES=<raw batches>  TRACES=<out dir>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
py=/home/bany/bin/python3
$py analysis/gen_shard_traces.py --cache $CACHE/d1 --out $TRACES --mode shard --alias 8 --sample 8 &
$py analysis/gen_shard_traces.py --cache $CACHE/pl_aff2 --out $TRACES --mode shard --alias 8 --sample 8 &
$py analysis/gen_step_traces.py --cache $CACHE/pl_aff2 --out $TRACES --mode mesi_rmw --warm-write --flush-mib 32 --sample 8 &
$py analysis/gen_shard_traces.py --cache $CACHE/d1 --out $TRACES --mode shard_sq --alias 8 --sample 8 --batches $BATCHES &
wait
