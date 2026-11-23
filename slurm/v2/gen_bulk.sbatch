#!/usr/bin/env bash
#SBATCH --job-name=gen-bulk
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=00:40:00
#SBATCH --output=logs/gen-bulk-%j.log
# Bulk-issue traces (tag suffix ig): requests 10 ticks apart, sharded rows
# round-robin over the owners. For validating analysis/linkmodel.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3; A=analysis
D=/scratch/engin_root/engin1/bany/dlrm_work; C=$D/criteo/cache; T=$D/traces/step
ig="--issue-gap 10"
$py $A/gen_shard_traces.py --cache $C/d1 --out $T --mode shard --alias 8 --interleave $ig --tag shard-d1-x64ig &
$py $A/gen_shard_traces.py --cache $C/pl_aff2 --out $T --mode shard --alias 8 --interleave $ig --tag shard-pl_aff2-x64ig &
$py $A/gen_shard_traces.py --cache $C/pl_aff2 --out $T --mode shard_mig --alias 8 --interleave $ig --tag shard_mig-pl_aff2-x64ig &
$py $A/gen_step_traces.py --cache $C/d1 --out $T --mode unc_agg $ig --tag unc_agg-d1-x64ig &
$py $A/gen_step_traces.py --cache $C/pl_aff2 --out $T --mode comm --keep --keep-window 16 $ig --tag comm_keep-pl_aff2kw16-x64ig &
$py $A/gen_step_traces.py --cache $C/pl_aff2 --out $T --mode mesi_rmw --warm-write $ig --tag mesi_ww-pl_aff2-x64ig &
wait
