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

source "${CXL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/env.sh"
cxl_load_build_modules

TRACE="${TRACE:?Set TRACE to an absolute CSV packet trace path}"
TRACE2="${TRACE2:-}"
NAME="${NAME:-trace}"
MAX_OUTSTANDING="${MAX_OUTSTANDING:-32}"
if [[ ! "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo 'NAME must contain only letters, digits, dots, underscores, or hyphens.' >&2
    exit 1
fi
if [[ "$TRACE" != /* || ( -n "$TRACE2" && "$TRACE2" != /* ) || "$RESULT_ROOT" != /* ]]; then
    echo 'TRACE, TRACE2, and RESULT_ROOT must be absolute paths.' >&2
    exit 1
fi
OUTDIR="$RESULT_ROOT/${NAME}-${SLURM_JOB_ID}"

[[ -x "$GEM5" ]] || { echo "missing $GEM5" >&2; exit 1; }
[[ -f "$TRACE" ]] || { echo "missing trace $TRACE" >&2; exit 1; }
[[ -z "$TRACE2" || -f "$TRACE2" ]] || { echo "missing trace $TRACE2" >&2; exit 1; }
mkdir -p "$RESULT_ROOT"
mkdir "$OUTDIR"
# The CSVTRACE config parser tokenizes on whitespace and cannot quote filenames.
if [[ "$OUTDIR" =~ [[:space:]] || "$TRACE" =~ [[:space:]] || "$TRACE2" =~ [[:space:]] ]]; then
    echo 'Trace and output paths must not contain whitespace.' >&2
    exit 1
fi

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

"$GEM5" -d "$OUTDIR" \
    "$CXL_PROJECT_ROOT/configs/cxl-trace.py" \
    --cpu-config "$OUTDIR/cpu-idle.cfg" \
    "${device_args[@]}" \
    --max-outstanding "$MAX_OUTSTANDING"

echo "output: $OUTDIR"
