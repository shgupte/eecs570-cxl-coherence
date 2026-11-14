#!/usr/bin/env bash
# Submit one gem5 run of an 8-GPU trace dir made by gen_ngpu_traces.py.
# usage: analysis/submit_ngpu.sh <prefix> <trace_dir> <config_tag> [extra gem5 args...]
# env: L2LAT (256), L1 (1MiB), L2 (64MiB), MAX_OUTSTANDING (32), TIME (02:00:00), PACK_FILE (append the run to a list instead of submitting)
set -euo pipefail
prefix=$1 dir=$2 cfg=$3; shift 3
traces="$(ls "$dir"/gpu*.csv | sort -V) $(ls "$dir"/dma*.csv 2>/dev/null | sort -V || true)"
extra="--l1d-size ${L1:-1MiB} --l2-size ${L2:-64MiB} --l2-latency-cycles ${L2LAT:-256} $*"
name="$prefix-$(basename "$dir")--$cfg"
if [[ -n "${PACK_FILE:-}" ]]; then   # queue for slurm/v2/pack_cxl_trace.sbatch instead of one job per run
    printf '%s\t%s\t%s\t%s\n' "$name" "${MAX_OUTSTANDING:-32}" "$extra" "$(echo $traces)" >> "$PACK_FILE"
    exit 0
fi
TRACES="$(echo $traces)" MAX_OUTSTANDING="${MAX_OUTSTANDING:-32}" sbatch --parsable --time="${TIME:-02:00:00}" \
    --export=ALL,NAME="$name",EXTRA_ARGS="$extra" slurm/v2/run_cxl_trace.sbatch
