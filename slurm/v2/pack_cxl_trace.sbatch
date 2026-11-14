#!/usr/bin/env bash
#SBATCH --job-name=cxl-pack
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=60G --time=03:00:00
#SBATCH --output=logs/cxl-pack-%j.log
# Run many gem5 trace replays inside one job (the class account runs 2 jobs).
# LIST = file of lines "name<TAB>max_outstanding<TAB>gem5 args<TAB>traces"
# (written by analysis/submit_ngpu.sh with PACK_FILE set); PAR = parallel runs.
# Each run gets id <jobid><3-digit index> and its own logs/cxl-trace-<id>.log.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
i=0
while IFS=$'\t' read -r name mo extra traces; do
    id=$(printf '%s%03d' "$SLURM_JOB_ID" "$i"); i=$((i + 1))
    ( SLURM_JOB_ID=$id NAME="$name" MAX_OUTSTANDING="$mo" EXTRA_ARGS="$extra" TRACES="$traces" \
        bash slurm/v2/run_cxl_trace.sbatch > "logs/cxl-trace-$id.log" 2>&1 || echo "FAILED $name" ) &
    while (( $(jobs -rp | wc -l) >= ${PAR:-16} )); do wait -n; done
done < "$LIST"
wait
echo "done $i runs"
