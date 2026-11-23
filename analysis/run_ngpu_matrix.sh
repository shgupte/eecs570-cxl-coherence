#!/usr/bin/env bash
# Full MESI-vs-alternatives matrix on 8-GPU Criteo traces (gen_ngpu_traces.py).
# usage (repo root): HOT=<hot_rows.npz> TRACES=<dir> PREFIX=ng analysis/run_ngpu_matrix.sh
# then:  python3 analysis/summarize_ngpu.py $RESULT_ROOT $PREFIX logs
set -euo pipefail
HOT="${HOT:?}"; TR="${TRACES:?}"; P="${PREFIX:-ng}"
MODES="mesi uncached swflush swinv update"
gen() {  # gen <mode> <args...>; echoes trace dir
    python3 analysis/gen_ngpu_traces.py --hot "$HOT" --out "$TR" --mode "$@" | awk '{print $1}'
}
sub() { analysis/submit_ngpu.sh "$P" "$TR/$1" "${@:2}" >/dev/null || echo "submit failed: $*" >&2; }

for m in $MODES; do
    for k in 1 2 4 8; do t=$(gen $m --stale $k); sub $t base; done          # A core
    for k in 1 4; do
        t=$(gen $m --stale $k)
        L2LAT=128 sub $t lat128; L2LAT=512 sub $t lat512                    # C CXL latency
        sub $t bw8 --link-bw 8; sub $t bw64 --link-bw 64                     # D link bandwidth
        MAX_OUTSTANDING=128 sub $t mlp128                                    # E memory-level parallelism
        t2=$(gen $m --stale $k --rows 2048 --alias 4); sub $t2 base          # F larger hot set
        t3=$(gen $m --stale $k --first-batch 16); sub $t3 base               # G other batches
        for g in 2 4; do t4=$(gen $m --stale $k --gpus $g); sub $t4 base; done   # H GPU count
        if [[ $m != uncached ]]; then
            for p in 0.5 0.25 0.1; do t5=$(gen $m --stale $k --write-frac $p); sub $t5 base; done  # B
            for p in 0.25 0.1; do t5=$(gen $m --stale $k --write-frac $p); sub $t5 bw64 --link-bw 64; done
        fi
    done
done
echo "submitted; jobs: $(squeue -u "$USER" -h | wc -l)"
