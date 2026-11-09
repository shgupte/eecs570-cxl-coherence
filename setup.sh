#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Run the build in a compute allocation: mkdir -p logs; sbatch slurm/build_simcxl.sbatch' >&2
    exit 1
fi
cxl_load_build_modules

base_commit=edddc2054bcdafdc7537b20c99605f2181bda9dc
patch_file="$CXL_PROJECT_ROOT/patches/simcxl-csv-two-accel.patch"
mkdir -p "$CXL_WORK_ROOT" "$(dirname -- "$SIMCXL_DIR")"
# Reject concurrent setup/build jobs targeting the same work directory.
lock_dir="$CXL_WORK_ROOT/.setup-lock"
if ! mkdir "$lock_dir"; then
    echo "Another setup may be running: $lock_dir (remove only if no job is using it)." >&2
    exit 1
fi
trap 'rmdir "$lock_dir"' EXIT

if [[ ! -e "$SIMCXL_DIR" ]]; then
    git init "$SIMCXL_DIR"
    git -C "$SIMCXL_DIR" remote add origin https://github.com/TianheMICALab/SimCXL.git
    git -C "$SIMCXL_DIR" fetch --depth 1 origin "$base_commit"
    git -C "$SIMCXL_DIR" checkout --detach "$base_commit"
fi

if [[ "$(git -C "$SIMCXL_DIR" rev-parse --show-toplevel)" != "$(cd "$SIMCXL_DIR" && pwd -P)" ]]; then
    echo "SIMCXL_DIR must point to the root of a simulator checkout: $SIMCXL_DIR" >&2
    exit 1
fi
if ! git -C "$SIMCXL_DIR" merge-base --is-ancestor "$base_commit" HEAD; then
    echo "Expected a SimCXL checkout based on $base_commit. Your checkout was preserved." >&2
    echo 'Use a separate pinned checkout for this baseline setup.' >&2
    exit 1
fi

# The marker allows later source edits without reapplying or undoing the patch.
marker="$(git -C "$SIMCXL_DIR" rev-parse --absolute-git-dir)/cxl-project-patch.sha256"
patch_hash="$(sha256sum "$patch_file" | cut -d ' ' -f 1)"
if [[ -f "$marker" ]]; then
    if [[ "$(cat "$marker")" != "$patch_hash" ]]; then
        echo 'The project patch changed. Use a new checkout; existing edits were preserved.' >&2
        exit 1
    fi
elif git -C "$SIMCXL_DIR" apply --reverse --check "$patch_file" 2>/dev/null; then
    printf '%s\n' "$patch_hash" > "$marker"
elif [[ -z "$(git -C "$SIMCXL_DIR" status --porcelain)" ]] && \
     git -C "$SIMCXL_DIR" apply --check "$patch_file"; then
    git -C "$SIMCXL_DIR" apply "$patch_file"
    printf '%s\n' "$patch_hash" > "$marker"
else
    echo 'Cannot safely apply the patch. Existing simulator changes were preserved.' >&2
    exit 1
fi

if [[ ! -x "$CXL_VENV/bin/python" ]]; then
    python3 -m venv "$CXL_VENV"
fi
source "$CXL_VENV/bin/activate"
python -m pip install -r "$CXL_PROJECT_ROOT/requirements-build.txt"

cd "$SIMCXL_DIR"
# This job builds the baseline protocol configuration on each invocation.
scons defconfig build/X86_CXL_MESI build_opts/X86
scons setconfig build/X86_CXL_MESI RUBY_PROTOCOL_CXL_MESI_TWO_LEVEL=y
scons build/X86_CXL_MESI/gem5.opt -j"${SLURM_CPUS_PER_TASK:-8}"

{
    date -u
    git rev-parse HEAD
    git diff --stat
    printf 'Project patch SHA256: %s\n' "$patch_hash"
    python --version
    python -m pip freeze
    gcc --version
    module list 2>&1
} > "$CXL_WORK_ROOT/build-environment.txt"
echo "Built: $SIMCXL_DIR/build/X86_CXL_MESI/gem5.opt"
