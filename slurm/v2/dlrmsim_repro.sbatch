#!/usr/bin/env bash
#SBATCH --job-name=dlrmsim-repro
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dlrmsim-repro-%j.log
set -euo pipefail
module load python/3.10.4 pytorch/2.0.1
cd ~/eecs570-cxl-coherence/dlrm-sim
DS=/scratch/engin_root/engin1/bany/dlrm_datasets/2021/fbgemm_t856_bs65536_0.pt.gz
OUT=/scratch/engin_root/engin1/bany/dlrmsim_repro
python main.py --dataset $DS --output-dir $OUT/torchrec
python main.py --dataset $DS --model coherent --protocol vi --output-dir $OUT/vi
