#!/usr/bin/env bash
#SBATCH --job-name=linkmodel-host
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00
#SBATCH --output=logs/linkmodel-host-%j.log
# Host-DRAM hybrid (rows in server memory, bulk PCIe) against the CXL pool,
# full batch. host_bw = what the host side can serve per direction, total.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=/scratch/engin_root/engin1/bany/pylib
py=/home/bany/bin/python3
C=/scratch/engin_root/engin1/bany/dlrm_work/criteo
CXL="--cxl-req 16 --cxl-data 85 --cxl-ack 16"
for c in d1 pl_aff2; do for hb in 50 100 200 504; do
  echo "## host hybrid cache=$c gpu=63 host_bw=$hb"
  $py analysis/linkmodel.py --sample 1 --cache $C/cache/$c --designs comm,comm_keep --gpu-bw 63 --pool-bw $hb --pool-bulk
done; done
for p in 4 8; do
  echo "## cxl pool cache=pl_aff2 gpu=63 ports=$p"
  $py analysis/linkmodel.py --sample 1 --cache $C/cache/pl_aff2 --designs unc_agg,comm_keep,mesi_comm --gpu-bw 63 --pool-bw $((63 * p)) $CXL
done
echo "## cxl pool, no placement"
$py analysis/linkmodel.py --sample 1 --cache $C/cache/d1 --designs comm,comm_keep,mesi_comm --gpu-bw 63 --pool-bw 504 $CXL
