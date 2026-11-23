#!/usr/bin/env bash
# Checks on top of run_port_matrix.sh (traces from slurm/v2/prep_checks.sbatch):
# sharding with row migration, software keep window, second step window
# (w16), second Criteo day (d2). Same flags as run_port_matrix.sh.
# usage: PACK_FILE=runs.tsv TRACES=<dir> analysis/run_port_checks.sh
set -euo pipefail
TR="${TRACES:?}"; : "${PACK_FILE:?}"
PM="--pool-mediated --pool-mediated-gets"
sub() { L2=64MiB analysis/submit_ngpu.sh sp "$@"; }
pool() {
    local t=$1 s=$2; shift 2
    for p in ${PORTS:-1 8}; do
        if [[ $t == mesi* ]]; then
            L1=1MiB sub "$TR/$t" "fwd-p$p$s" --topology pool --pool-ports "$p" "$@"
            L1=1MiB sub "$TR/$t" "pm-p$p$s" --topology pool --pool-ports "$p" $PM "$@"
            L1=1MiB sub "$TR/$t" "coup-p$p$s" --topology pool --pool-ports "$p" $PM --commutative "$@"
        else
            L1=64KiB sub "$TR/$t" "p$p$s" --topology pool --pool-ports "$p" --commutative "$@"
        fi
    done
}
shard() { local t=$1 s=$2; shift 2; L1=64KiB sub "$TR/$t" "sh$s" --topology sharded --commutative "$@"; }
# row migration (day 1, window 2-13) with sensitivity
shard shard_mig-d1-x64 ""; shard shard_mig-pl_aff2-x64 ""
shard shard_mig-pl_aff2-x64 -bw8 --gpu-bw 8; shard shard_mig-pl_aff2-x64 -bw64 --gpu-bw 64
L2LAT=128 shard shard_mig-pl_aff2-x64 -lat128; L2LAT=1024 shard shard_mig-pl_aff2-x64 -lat1024
# software keep window
for k in 4 8 16; do pool comm_keep-pl_aff2kw$k-x64 ""; done
# second window and second day
for v in "d1w16 pl_aff2w16" "d2 pl_aff2d2"; do
    set -- $v
    pool unc_agg-$1-x64 ""; pool comm_keep-$2-x64 ""; pool mesi_ww-$2-x64 ""
    shard shard-$1-x64 ""; shard shard-$2-x64 ""; shard shard_mig-$2-x64 ""; shard shard_sq-$1-x64 ""
done
