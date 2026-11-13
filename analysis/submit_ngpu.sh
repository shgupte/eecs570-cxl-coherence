#!/usr/bin/env bash
# Submit one gem5 run of an 8-GPU trace dir made by gen_ngpu_traces.py.
# usage: analysis/submit_ngpu.sh <prefix> <trace_dir> <config_tag> [extra gem5 args...]
# env: L2LAT (256), L1 (1MiB), L2 (64MiB), MAX_OUTSTANDING (32)
set -euo pipefail
prefix=$1 dir=$2 cfg=$3; shift 3
traces="$(ls "$dir"/gpu*.csv | sort -V) $(ls "$dir"/dma*.csv 2>/dev/null | sort -V || true)"
TRACES="$(echo $traces)" MAX_OUTSTANDING="${MAX_OUTSTANDING:-32}" sbatch --parsable --time=02:00:00 \
    --export=ALL,NAME="$prefix-$(basename "$dir")--$cfg",EXTRA_ARGS="--l1d-size ${L1:-1MiB} --l2-size ${L2:-64MiB} --l2-latency-cycles ${L2LAT:-256} $*" \
    slurm/v2/run_cxl_trace.sbatch
