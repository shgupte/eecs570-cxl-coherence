#!/usr/bin/env bash
# Run on Great Lakes: submit, wait, then show the saved summary.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs results

echo "Submitting sharded-memory job and waiting for completion..."
status=0
submission=$(sbatch --wait --parsable slurm/run_sharded_memory.sbatch) || status=$?
job_id=${submission%%;*}

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Slurm did not return a job ID (exit status $status)." >&2
    exit 1
fi

result_dir="results/sharded-memory-${job_id}"
log="logs/sharded-memory-${job_id}.log"
echo "Job: $job_id"

if (( status != 0 )); then
    echo "Job failed (exit status $status). Log: $log" >&2
    if [[ -f "$log" ]]; then
        tail -n 60 "$log" >&2
    fi
    exit "$status"
fi

if [[ ! -s "$result_dir/summary.txt" ]]; then
    echo "Job returned success but no summary was saved. Check $log" >&2
    exit 1
fi

cat "$result_dir/summary.txt"
echo "Saved: $result_dir/summary.txt"
echo "Full log: $log"
