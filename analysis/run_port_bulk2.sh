#!/usr/bin/env bash
# Bulk-transfer follow-ups (window larger than a GPU's phase):
#  eq    pool with the same total link bandwidth as sharding (8 x 16 B/cycle):
#        GPU ports of b B/cycle and a pool link of 128 - 8b
#  rob   second step window, second day, 8x scale
#  bw8   half-speed ports
# usage: PACK_FILE=runs.tsv TRACES=<dir> analysis/run_port_bulk2.sh
set -euo pipefail
TR="${TRACES:?}"; : "${PACK_FILE:?}"
PM="--pool-mediated --pool-mediated-gets"
sub() { analysis/submit_ngpu.sh sb "$@"; }
unc() { local t=$1 c=$2; shift 2; L1=64KiB sub "$TR/$t" "$c" --topology pool --commutative "$@"; }
coup() { local t=$1 c=$2; shift 2; sub "$TR/$t" "coup-$c" --topology pool $PM --commutative "$@"; }
shard() { local t=$1 c=$2; shift 2; L1=64KiB sub "$TR/$t" "sh$c" --topology sharded --commutative "$@"; }
export MAX_OUTSTANDING=16384 L2=64MiB
for b in 8 10 12; do
    a="--gpu-bw $b --pool-bw $((128 - 8 * b))"
    unc unc_agg-d1-x64 "eq$b" $a; unc comm_keep-pl_aff2kw16-x64 "eq$b" $a
    L1=1MiB coup mesi_ww-pl_aff2-x64 "eq$b" $a
done
for v in "d1w16 pl_aff2w16" "d2 pl_aff2d2"; do
    set -- $v
    unc unc_agg-$1-x64 p8-mo --pool-ports 8; unc comm_keep-$2-x64 p8-mo --pool-ports 8
    L1=1MiB coup mesi_ww-$2-x64 p8-mo --pool-ports 8
    shard shard-$1-x64 -mo; shard shard-$2-x64 -mo; shard shard_mig-$2-x64 -mo
done
shard shard-pl_aff2-x64 -mo-bw8 --gpu-bw 8; shard shard_mig-pl_aff2-x64 -mo-bw8 --gpu-bw 8
unc comm_keep-pl_aff2kw16-x64 p8-mo-bw8 --pool-ports 8 --gpu-bw 8
L1=1MiB coup mesi_ww-pl_aff2-x64 p8-mo-bw8 --pool-ports 8 --gpu-bw 8
export MAX_OUTSTANDING=131072 L2=512MiB
unc unc_agg-d1-x8 p8-mo --pool-ports 8; unc comm_keep-pl_aff2-x8 p8-mo --pool-ports 8
L1=8MiB coup mesi_ww-pl_aff2-x8 p8-mo --pool-ports 8
shard shard-d1-x8 -mo; shard shard-pl_aff2-x8 -mo; shard shard_mig-pl_aff2-x8 -mo
