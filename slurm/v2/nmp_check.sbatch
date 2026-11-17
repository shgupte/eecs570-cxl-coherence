#!/usr/bin/env bash
#SBATCH --job-name=nmp
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00
#SBATCH --output=logs/nmp-%j.log
# Pool with near-memory pooling and gradient apply, against pooled all-to-all.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib:analysis
py=/home/bany/bin/python3; D=/scratch/engin_root/engin1/bany/dlrm_work/criteo
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
m() { $py analysis/linkmodel.py --sample 1 --batches $D/batches/d1 "$@" | cut -d, -f1-9; }
for c in "mh_d1 --bag-sizes" "d1"; do
  set -- $c; cache=$1; shift
  for p in 1 2 4 8; do echo "## $cache pool gpu=63 ports=$p"; m --cache $D/cache/$cache "$@" --designs nmp,nmp_dedup --gpu-bw 63 --pool-bw $((63 * p)) $CXL; done
  echo "## $cache pool gpu=121 ports=8"; m --cache $D/cache/$cache "$@" --designs nmp,nmp_dedup --gpu-bw 121 --pool-bw 968 $CXL
  for nv in 300 50 25; do echo "## $cache sharded pooled nv=$nv"; m --cache $D/cache/$cache "$@" --designs pooled,pooled_dedup --nv-bw $nv; done
done
