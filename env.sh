#!/usr/bin/env bash
# Source this file to share paths between interactive shells and batch jobs.
export CXL_PROJECT_ROOT
CXL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$CXL_PROJECT_ROOT/env.local.sh" ]]; then
    source "$CXL_PROJECT_ROOT/env.local.sh"
fi

export CXL_WORK_ROOT="${CXL_WORK_ROOT:-$CXL_PROJECT_ROOT/.work}"
export SIMCXL_DIR="${SIMCXL_DIR:-$CXL_WORK_ROOT/sources/SimCXL}"
export CXL_VENV="${CXL_VENV:-$CXL_WORK_ROOT/venv-build}"
export GEM5="${GEM5:-$SIMCXL_DIR/build/X86_CXL_MESI/gem5.opt}"
export RESULT_ROOT="${RESULT_ROOT:-$CXL_WORK_ROOT/sim-results}"
export OUTPUT_DIR="${OUTPUT_DIR:-$CXL_WORK_ROOT/trace-v2}"
export OUTPUT="${OUTPUT:-$CXL_WORK_ROOT/inspection.json}"
export CXL_SHARED_ROOT="${CXL_SHARED_ROOT:-/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shared_data/eecs570-cxl-coherence}"
export DATASET="${DATASET:-$CXL_SHARED_ROOT/datasets/fbgemm_t856_bs65536_0.pt.gz}"
export CXL_GCC_MODULE="${CXL_GCC_MODULE:-gcc/11.2.0}"
export CXL_PYTHON_MODULE="${CXL_PYTHON_MODULE:-python/3.11.5}"
export CXL_TRACE_PYTHON_MODULE="${CXL_TRACE_PYTHON_MODULE:-python/3.10.4}"
export CXL_TORCH_MODULE="${CXL_TORCH_MODULE:-pytorch/2.0.1}"

cxl_load_build_modules() {
    module purge
    module load "$CXL_GCC_MODULE" "$CXL_PYTHON_MODULE"
}

cxl_load_trace_modules() {
    # Use the cluster PyTorch environment, independently of the build venv.
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo 'Start a fresh shell without an activated venv for dataset jobs.' >&2
        return 1
    fi
    module purge
    module load "$CXL_TRACE_PYTHON_MODULE" "$CXL_TORCH_MODULE"
}
