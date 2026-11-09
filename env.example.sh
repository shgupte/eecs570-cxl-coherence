# Copy to env.local.sh and edit. This file is sourced by Bash.
# Each person should use their own clone and work directory.
CXL_WORK_ROOT="/scratch/eecs570f26s001_class_root/eecs570f26s001_class/${USER}/cxl-coherence"

# Shared read-only inputs; defaults are provided by env.sh.
# CXL_SHARED_ROOT="/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shared_data/eecs570-cxl-coherence"
# DATASET="$CXL_SHARED_ROOT/datasets/fbgemm_t856_bs65536_0.pt.gz"

# Existing personal checkouts can be selected explicitly:
# SIMCXL_DIR="$HOME/cxl_gl_agent/sources/SimCXL"
# RESULT_ROOT="$CXL_WORK_ROOT/sim-results"

# Defaults match the original cluster jobs. Override if modules change.
# CXL_GCC_MODULE="gcc/11.2.0"
# CXL_PYTHON_MODULE="python/3.11.5"
# CXL_TRACE_PYTHON_MODULE="python/3.10.4"
# CXL_TORCH_MODULE="pytorch/2.0.1"
