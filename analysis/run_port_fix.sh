#!/usr/bin/env bash
# After the Downgrade fix (patches/simcxl-v5-downgrade.patch):
#  reg   regression: designs the fix must not change
#  coup  MESI + commutative updates again (window 32 and bulk)
#  ig    bulk-issue traces (gen_bulk.sbatch) for validating linkmodel.py
#  x8    8x scale, bulk
# usage: PACK_FILE=runs.tsv TRACES=<dir> analysis/run_port_fix.sh
set -euo pipefail
TR="${TRACES:?}"; : "${PACK_FILE:?}"
PM="--pool-mediated --pool-mediated-gets"
sub() { analysis/submit_ngpu.sh sf "$@"; }
unc() { local t=$1 c=$2; shift 2; L1=64KiB sub "$TR/$t" "$c" --topology pool --commutative "$@"; }
coup() { local t=$1 c=$2; shift 2; sub "$TR/$t" "coup-$c" --topology pool $PM --commutative "$@"; }
shard() { local t=$1 c=$2; shift 2; L1=64KiB sub "$TR/$t" "sh$c" --topology sharded --commutative "$@"; }
export L2=64MiB L1=1MiB
MAX_OUTSTANDING=32 unc comm_keep-pl_aff2kw16-x64 p8 --pool-ports 8
MAX_OUTSTANDING=32 sub "$TR/mesi_ww-pl_aff2-x64" pm-p8 --topology pool --pool-ports 8 $PM
for p in 1 2 4 8; do
    MAX_OUTSTANDING=32 coup mesi_ww-pl_aff2-x64 p$p --pool-ports $p
    MAX_OUTSTANDING=16384 coup mesi_ww-pl_aff2-x64 p$p-mo --pool-ports $p
done
MAX_OUTSTANDING=32 coup mesi_ww-d1-x64 p8 --pool-ports 8
for v in pl_aff2w16 pl_aff2d2; do
    MAX_OUTSTANDING=32 coup mesi_ww-$v-x64 p8 --pool-ports 8; MAX_OUTSTANDING=16384 coup mesi_ww-$v-x64 p8-mo --pool-ports 8
done
MAX_OUTSTANDING=16384 coup mesi_ww-pl_aff2-x64 eq8 --gpu-bw 8 --pool-bw 64
export MAX_OUTSTANDING=16384
for p in 1 8; do
    unc unc_agg-d1-x64ig p$p-mo --pool-ports $p; unc comm_keep-pl_aff2kw16-x64ig p$p-mo --pool-ports $p
    coup mesi_ww-pl_aff2-x64ig p$p-mo --pool-ports $p
done
shard shard-d1-x64ig -mo; shard shard-pl_aff2-x64ig -mo; shard shard_mig-pl_aff2-x64ig -mo
export MAX_OUTSTANDING=131072 L2=512MiB
unc unc_agg-d1-x8 p8-mo --pool-ports 8; unc comm_keep-pl_aff2-x8 p8-mo --pool-ports 8
L1=8MiB coup mesi_ww-pl_aff2-x8 p8-mo --pool-ports 8
shard shard-d1-x8 -mo; shard shard-pl_aff2-x8 -mo; shard shard_mig-pl_aff2-x8 -mo
