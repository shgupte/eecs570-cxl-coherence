#!/usr/bin/env bash
#SBATCH --job-name=cxl-trace
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cxl-trace-%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load gcc/11.2.0 python/3.11.5

TRACE="${TRACE:?Set TRACE to an absolute CSV packet trace path}"
TRACE2="${TRACE2:-}"
TRACE3="${TRACE3:-}"
NAME="${NAME:-trace}"
MAX_OUTSTANDING="${MAX_OUTSTANDING:-32}"
GEM5="${GEM5:-/scratch/engin_root/engin1/bany/simcxl-baseline/build/X86_CXL_MESI/gem5.opt}"
RESULT_ROOT="${RESULT_ROOT:-/scratch/engin_root/engin1/bany/dlrm_work/sim-results}"
OUTDIR="$RESULT_ROOT/${NAME}-${SLURM_JOB_ID}"

[[ -x "$GEM5" ]] || { echo "missing $GEM5" >&2; exit 1; }
[[ -f "$TRACE" ]] || { echo "missing trace $TRACE" >&2; exit 1; }
[[ -z "$TRACE2" || -f "$TRACE2" ]] || { echo "missing trace $TRACE2" >&2; exit 1; }
mkdir -p "$OUTDIR"

cat > "$OUTDIR/cpu-idle.cfg" <<'EOF'
STATE 0 1000000000000 IDLE
INIT 0
TRANSITION 0 0 1
EOF

cat > "$OUTDIR/device-trace.cfg" <<EOF
STATE 0 0 CSVTRACE $TRACE 0
STATE 1 1000000000 IDLE
STATE 2 1 EXIT
INIT 0
TRANSITION 0 1 1
TRANSITION 1 2 1
TRANSITION 2 2 1
EOF

device_args=(--device-config "$OUTDIR/device-trace.cfg")
if [[ -n "$TRACE2" ]]; then
    cat > "$OUTDIR/device-trace-1.cfg" <<EOF
STATE 0 0 CSVTRACE $TRACE2 0
STATE 1 1000000000 IDLE
STATE 2 1 EXIT
INIT 0
TRANSITION 0 1 1
TRANSITION 1 2 1
TRANSITION 2 2 1
EOF
    device_args+=(--device-config "$OUTDIR/device-trace-1.cfg")
fi

if [[ -n "$TRACE3" ]]; then
    sed "s#$TRACE2#$TRACE3#" "$OUTDIR/device-trace-1.cfg" > "$OUTDIR/device-trace-2.cfg"
    device_args+=(--device-config "$OUTDIR/device-trace-2.cfg")
fi

"$GEM5" -d "$OUTDIR" \
    "$SLURM_SUBMIT_DIR/configs/cxl-trace.py" \
    --cpu-config "$OUTDIR/cpu-idle.cfg" \
    "${device_args[@]}" \
    --max-outstanding "$MAX_OUTSTANDING" ${EXTRA_ARGS:-}

echo "output: $OUTDIR"
