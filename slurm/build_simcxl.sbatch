#!/usr/bin/env bash
#SBATCH --job-name=cxl-build
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [[ ! -d sources/SimCXL/.git ]]; then
    git clone https://github.com/TianheMICALab/SimCXL.git sources/SimCXL
fi
cd sources/SimCXL
module load gcc/11.2.0 python/3.11.5
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Configure X86 with the CXL two-level MESI Ruby protocol
scons defconfig build/X86_CXL_MESI build_opts/X86
scons setconfig build/X86_CXL_MESI RUBY_PROTOCOL_CXL_MESI_TWO_LEVEL=y
scons build/X86_CXL_MESI/gem5.opt -j"${SLURM_CPUS_PER_TASK:-8}"
