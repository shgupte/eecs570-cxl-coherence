#!/usr/bin/env bash
# Generate the coherence-baseline traces and submit one gem5 job per run.
# Run from the repo root on a Great Lakes login node:
#   HOT=<path>/hot_rows.npz analysis/run_coherence_sweep.sh
# Optional env: TRACES (trace dir), LATS ("256" or "128 256 512"),
#               KS ("1 2 4 8"), PREFIX (run-name prefix, default "sel"),
#               GEM5 / RESULT_ROOT (passed through to slurm/v2/run_cxl_trace.sbatch).
set -euo pipefail

HOT="${HOT:?set HOT to hot_rows.npz from reuse_analysis.py}"
TRACES="${TRACES:-$PWD/traces/coherence}"
LATS="${LATS:-256}"
KS="${KS:-1 2 4 8}"
PREFIX="${PREFIX:-sel}"
GEN="python3 analysis/gen_multistep_traces.py --hot $HOT --out $TRACES --rows 1024 --steps 16"

tags=()
for k in $KS; do
    $GEN --mode shared --stale "$k"
    $GEN --mode uncached --alias 4 --warm --stale "$k"
    $GEN --mode swflush --alias 4 --warm --stale "$k"
    s=$([[ $k == 1 ]] && echo "" || echo "-s$k")
    tags+=("shared$s" "uncached$s-a4-w" "swflush$s-a4-w")
done

mkdir -p logs
for lat in $LATS; do
    for t in "${tags[@]}"; do
        sbatch --parsable --time=02:00:00 \
            --export=ALL,NAME=$PREFIX-$t-1MiB-l2lat$lat,TRACE=$TRACES/$t-accel0.csv,TRACE2=$TRACES/$t-accel1.csv,TRACE3=$TRACES/$t-owner.csv,EXTRA_ARGS="--l1d-size 1MiB --l2-size 8MiB --l2-latency-cycles $lat" \
            slurm/v2/run_cxl_trace.sbatch
    done
done
echo "when the jobs finish:"
echo "  python3 analysis/summarize_sweep.py \${RESULT_ROOT:-/scratch/engin_root/engin1/bany/dlrm_work/sim-results} $PREFIX logs"
