#!/usr/bin/env bash
#SBATCH --job-name=sharded-collective
#SBATCH --account=eecs570f26s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sharded-collective-%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results traces

module purge
module load python/3.10.4 pytorch/2.0.1

DATASET="/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shgupte/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_0.pt.gz"
TRACE_DIR="$SLURM_SUBMIT_DIR/traces/sharded-collective-128x4"

python scripts/generate_sharded_lookups.py "$DATASET" "$TRACE_DIR" \
    --samples 128 \
    --tables 4 \
    --collective \
    --phase-gap 0 \
    --drain-ticks 1000000000

module purge
module load gcc/11.2.0 python/3.11.5

GEM5="/home/shgupte/cxl_gl_agent/sources/SimCXL/build/X86_CXL_MESI/gem5.opt"
RESULT_DIR="results/sharded-collective-128x4-${SLURM_JOB_ID}"

TRACE_DIR="$TRACE_DIR" "$GEM5" \
    -d "$RESULT_DIR" \
    configs/sharded_memory.py

python3 scripts/summarize_sharded_memory.py "$RESULT_DIR" \
    | tee "$RESULT_DIR/summary.txt"

python3 - "$TRACE_DIR/manifest.json" <<'PY'
import json
import sys
manifest = json.loads(open(sys.argv[1]).read())
accelerators = manifest["accelerators"]
print("schedule", manifest["schedule"])
print(
    "gradient_start", manifest["gradient_start_tick"],
    "update_start", manifest["update_start_tick"],
    "last", manifest["last_tick"],
)
print(
    "id_sends", sum(item["id_sends"] for item in accelerators),
    "local_reads", sum(item["local_row_reads"] for item in accelerators),
    "pooled_results", sum(item["pooled_results"] for item in accelerators),
    "gradients", sum(item["gradients"] for item in accelerators),
    "row_writes", sum(item["row_writes"] for item in accelerators),
)
PY
