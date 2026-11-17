#!/usr/bin/env bash
#SBATCH --job-name=mh-prep
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=02:00:00
#SBATCH --output=logs/mh-prep-%j.log
# MLPerf-style multi-hot Criteo: per-step caches without and with placement,
# then the link model at full batch (one-hot next to it for reference).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3; A=analysis
D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
$py $A/sample_placement.py --batches $D/batches/d1 --out $D/cache/mh_d1 --policy contig --multihot --steps 32 | tail -n 2 &
$py $A/sample_placement.py --batches $D/batches/d1 --out $D/cache/mh_aff --policy affinity --cmax 512 --beta 2 --passes 0 --fast --multihot --steps 32 | tail -n 3 &
wait
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
DES=shard,shard_mig,unc_agg,comm_keep,mesi_pm,mesi_comm
for c in d1 pl_aff2 mh_d1 mh_aff; do
  ( echo "## $c"; $py $A/linkmodel.py --sample 1 --cache $D/cache/$c --designs $DES --gpu-bw 63 --pool-bw 504 --nv-bw 300 $CXL ) > logs/mh-model-$c.txt 2>&1 &
done; wait
cat logs/mh-model-*.txt
