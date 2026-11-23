#!/usr/bin/env bash
# Whole-step matrix (gen_step_traces.py): 6 designs x placements x configs.
# usage (repo root): CACHE=<dir with d1, pl_greedy, pl_aff2> TRACES=<dir> analysis/run_step_matrix.sh
# then: python3 analysis/summarize_step.py $RESULT_ROOT st logs --first-step 5
set -euo pipefail
C="${CACHE:?}"; TR="${TRACES:?}"; P="${PREFIX:-st}"
MODES=("unc_agg" "comm" "mesi_rmw" "mesi_agg" "unc_agg --keep" "comm --keep")
sub() {  # sub <trace dir> <cfg> [gem5 args]; L1 scale via S (1 or 8)
    local n; n=$(basename "$1"); local l1=$((512 * ${S:-1}))KiB
    [[ $n == mesi* ]] && l1=$((1 * ${S:-1}))MiB
    L1=$l1 L2=$((64 * ${S:-1}))MiB TIME=02:00:00 analysis/submit_ngpu.sh "$P" "$@" >/dev/null
}
for c in d1 pl_greedy pl_aff2; do
    for m in "${MODES[@]}"; do
        t=$TR/$(python3 analysis/gen_step_traces.py --cache "$C/$c" --out "$TR" --mode $m | awk '{print $1}')
        sub "$t" base
        [[ $c == pl_greedy ]] && continue
        L2LAT=128 sub "$t" lat128; L2LAT=512 sub "$t" lat512
        sub "$t" bw8 --link-bw 8; sub "$t" bw64 --link-bw 64
        MAX_OUTSTANDING=128 sub "$t" mlp128
        t8=$TR/$(python3 analysis/gen_step_traces.py --cache "$C/$c" --out "$TR" --mode $m --sample 8 | awk '{print $1}')
        S=8 sub "$t8" base
    done
done
