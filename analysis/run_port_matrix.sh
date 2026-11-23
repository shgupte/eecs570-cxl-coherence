#!/usr/bin/env bash
# Pool vs sharded all-to-all with explicit ports and real update messages.
# Needs SimCXL with patches/simcxl-v4-commutative.patch.
#   uncacheable designs (unc_agg, comm, shard...): --commutative, so a store
#     is one data message to the pool / the owner and one ack (no emulation
#     by write-back); traces from gen_step_traces.py / gen_shard_traces.py.
#     Their L1 is 64 KiB so that no line read through a private alias is
#     still cached when the alias is reused (ld_hit must be ~0).
#   MESI designs: traces from gen_step_traces.py --mode mesi_rmw --warm-write
#     fwd  = MESI_Two_Level as is (GPU-to-GPU forwarding)
#     pm   = CXL-style, every transfer through the pool
#     coup = pm + commutative updates of shared lines (COUP-style)
# usage: PACK_FILE=runs.tsv TRACES=<dir> analysis/run_port_matrix.sh
# then:  sbatch --mem=120G --export=ALL,LIST=runs.tsv,PAR=12,RESULT_ROOT=<dir> slurm/v2/pack_cxl_trace.sbatch
#        python3 analysis/summarize_step.py <dir> sp logs --first-step 5
set -euo pipefail
TR="${TRACES:?}"; : "${PACK_FILE:?}"; X="${X:-64}"; SC=$((64 / X))
PM="--pool-mediated --pool-mediated-gets"
sub() { L2=$((64 * SC))MiB analysis/submit_ngpu.sh sp "$@"; }
pool() {  # pool <trace tag> <cfg suffix> [gem5 args]; env PORTS
    local t=$1 s=$2; shift 2
    for p in ${PORTS:-1 2 4 8}; do
        if [[ $t == mesi* ]]; then
            L1=$((SC))MiB sub "$TR/$t" "fwd-p$p$s" --topology pool --pool-ports "$p" "$@"
            L1=$((SC))MiB sub "$TR/$t" "pm-p$p$s" --topology pool --pool-ports "$p" $PM "$@"
            L1=$((SC))MiB sub "$TR/$t" "coup-p$p$s" --topology pool --pool-ports "$p" $PM --commutative "$@"
        else
            L1=64KiB sub "$TR/$t" "p$p$s" --topology pool --pool-ports "$p" --commutative "$@"
        fi
    done
}
shard() {  # shard <trace tag> <cfg suffix> [gem5 args]
    local t=$1 s=$2; shift 2
    L1=64KiB sub "$TR/$t" "sh$s" --topology sharded --commutative "$@"
}
if [[ $X != 64 ]]; then   # scale check: headline designs only
    PORTS="1 8" pool unc_agg-d1-x$X ""; PORTS="1 8" pool comm_keep-pl_aff2-x$X ""
    PORTS="8" pool mesi_ww-pl_aff2-x$X ""
    shard shard-d1-x$X ""; shard shard-pl_aff2-x$X ""; shard shard_sq-d1-x$X ""
    exit 0
fi
for c in d1 pl_aff2; do
    for m in unc_agg unc_agg_keep comm comm_keep; do pool "$m-$c-x64" ""; done
    pool "mesi_ww-$c-x64" ""
    shard "shard-$c-x64" ""; shard "shard_sq-$c-x64" ""
done
# sensitivity: port bandwidth and hop latency
for t in unc_agg-d1-x64 comm_keep-pl_aff2-x64 mesi_ww-pl_aff2-x64; do
    PORTS="1 8" pool $t -bw8 --gpu-bw 8; PORTS="1 8" pool $t -bw64 --gpu-bw 64
    L2LAT=128 PORTS="1 8" pool $t -lat128; L2LAT=1024 PORTS="1 8" pool $t -lat1024
done
for t in shard-d1-x64 shard-pl_aff2-x64 shard_sq-d1-x64; do
    shard $t -bw8 --gpu-bw 8; shard $t -bw64 --gpu-bw 64
    L2LAT=128 shard $t -lat128; L2LAT=1024 shard $t -lat1024
done
