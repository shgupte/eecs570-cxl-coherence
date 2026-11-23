#!/usr/bin/env bash
#SBATCH --job-name=stale-train
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=80G --time=04:00:00
#SBATCH --output=logs/stale-train-%j.log
# CONFIGS="delay:lr_emb:seed ..." ; DATA=<npz from staleness_train.py prep>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
for c in $CONFIGS; do
    IFS=: read -r d lr seed <<<"$c"
    /home/bany/bin/python3 analysis/staleness_train.py train --data "$DATA" --delay "$d" --lr-emb "$lr" \
        --seed "$seed" --threads 4 --lr-dense "${LR_DENSE:-0.3}" > "logs/stale-$SLURM_JOB_ID-d$d-lr$lr-s$seed.log" 2>&1 &
done
wait
grep -h RESULT logs/stale-$SLURM_JOB_ID-*.log
