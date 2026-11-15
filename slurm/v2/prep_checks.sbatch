#!/usr/bin/env bash
#SBATCH --job-name=prep-checks
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=02:00:00
#SBATCH --output=logs/prep-checks-%j.log
# Traces for analysis/run_port_checks.sh: sharding with row migration, keep
# windows, a second step window (16-27) and a second Criteo day (d2).
# CRITEO=<dir with day=YYYY-MM-DD>  W=<work dir with batches/ cache/>  TRACES=<out dir>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3; A=analysis; C=$W/cache; B=$W/batches
if [[ ! -f $B/d2/b0.npz ]]; then
    $py $A/criteo_to_batches.py --day-dir "$CRITEO/day=${DAY2:-2015-03-10}" --out $B/d2 --batches 32
fi
n2=$(ls $B/d2 | wc -l)   # the day may have fewer than 32 batches on disk; the traces need 18
$py $A/sample_placement.py --batches $B/d2 --out $C/d2 --policy contig --steps $n2 | tail -n 1
$py $A/sample_placement.py --batches $B/d2 --out $C/pl_aff2d2 --policy affinity --cmax 512 --beta 2 --passes 0 --fast --steps $n2 | tail -n 2
$py $A/step_traffic.py $C/d2; $py $A/step_traffic.py $C/pl_aff2d2
gen() {  # gen <suffix> <contig cache> <placed cache> <batches> [--first N]
    local sfx=$1 c0=$2 c1=$3 b=$4; shift 4
    $py $A/gen_step_traces.py --cache $C/$c0 --out $TRACES --mode unc_agg --tag unc_agg-$c0$sfx-x64 "$@" &
    $py $A/gen_step_traces.py --cache $C/$c1 --out $TRACES --mode comm --keep --tag comm_keep-$c1$sfx-x64 "$@" &
    $py $A/gen_step_traces.py --cache $C/$c1 --out $TRACES --mode mesi_rmw --warm-write --tag mesi_ww-$c1$sfx-x64 "$@" &
    $py $A/gen_shard_traces.py --cache $C/$c0 --out $TRACES --mode shard --alias 8 --tag shard-$c0$sfx-x64 "$@" &
    $py $A/gen_shard_traces.py --cache $C/$c1 --out $TRACES --mode shard --alias 8 --tag shard-$c1$sfx-x64 "$@" &
    $py $A/gen_shard_traces.py --cache $C/$c1 --out $TRACES --mode shard_mig --alias 8 --tag shard_mig-$c1$sfx-x64 "$@" &
    $py $A/gen_shard_traces.py --cache $C/$c0 --out $TRACES --mode shard_sq --alias 8 --batches $b --tag shard_sq-$c0$sfx-x64 "$@" &
    wait
}
gen w16 d1 pl_aff2 $B/d1 --first 16
gen "" d2 pl_aff2d2 $B/d2
# day 1, window 2-13: migration and keep windows
$py $A/gen_shard_traces.py --cache $C/d1 --out $TRACES --mode shard_mig --alias 8 &
$py $A/gen_shard_traces.py --cache $C/pl_aff2 --out $TRACES --mode shard_mig --alias 8 &
for k in 4 8 16; do
    $py $A/gen_step_traces.py --cache $C/pl_aff2 --out $TRACES --mode comm --keep --keep-window $k --tag comm_keep-pl_aff2kw$k-x64 &
done
wait
