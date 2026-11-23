#!/usr/bin/env bash
# Bulk-transfer check: the same designs with a larger request window per GPU
# (MAX_OUTSTANDING). Real all-to-all sends one message per GPU pair, so the
# link latency is paid once, not once per 32 lines; a window larger than a
# GPU's phase models that. The pool designs get the same windows.
# usage: PACK_FILE=runs.tsv TRACES=<dir> analysis/run_port_bulk.sh
set -euo pipefail
TR="${TRACES:?}"; : "${PACK_FILE:?}"
PM="--pool-mediated --pool-mediated-gets"
sub() { L2=64MiB analysis/submit_ngpu.sh sb "$@"; }
for mo in ${WINDOWS:-256 16384}; do
    export MAX_OUTSTANDING=$mo
    for t in shard_sq-d1-x64 shard-d1-x64 shard-pl_aff2-x64 shard_mig-pl_aff2-x64; do
        L1=64KiB sub "$TR/$t" "sh-mo$mo" --topology sharded --commutative
    done
    for p in ${PORTS:-1 8}; do
        for t in unc_agg-d1-x64 unc_agg-pl_aff2-x64 comm_keep-pl_aff2kw16-x64; do
            L1=64KiB sub "$TR/$t" "p$p-mo$mo" --topology pool --pool-ports "$p" --commutative
        done
        L1=1MiB sub "$TR/mesi_ww-pl_aff2-x64" "coup-p$p-mo$mo" --topology pool --pool-ports "$p" $PM --commutative
    done
done
