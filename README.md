# Two-accelerator CXL coherence experiment

SimCXL comparison of two accelerators reading and updating the same DLRM embedding lines versus the same requests on disjoint lines.

This repo holds the patch, replay scripts, small traces, and saved results. Each person uses their own SimCXL checkout and build, while the large DLRM dataset can be shared read-only. No dataset download or guest images are needed to replay the committed traces.

## Great Lakes setup

After cloning this repository on Great Lakes, run from its root:

```bash
cp env.example.sh env.local.sh
# Edit env.local.sh: set your personal work directory and, optionally, DATASET.
mkdir -p logs
sbatch slurm/build_simcxl.sbatch
```

`env.local.sh` is a Bash file ignored by Git. The example places builds and outputs in your own directory under the class scratch space. Choose a directory you can write to; scratch is not a permanent backup. Without a local config, paths default to `.work/` inside this clone. Keep source changes in version control or backed up separately.

The build job runs `setup.sh`, which loads the compiler/Python modules, fetches the pinned simulator commit, applies the project patch, creates a build-only virtual environment, and compiles the CXL MESI simulator. Build tools are pinned in `requirements-build.txt`. The default modules, clean build, and litmus smoke test were verified on Great Lakes on November 9, 2026 (jobs 65725001 and 65725014). Environment details are recorded in `build-environment.txt` under your work directory.

Re-running the build reuses the checkout and compilation outputs. It does not reset or delete simulator edits. Once the patch is recorded, both uncommitted edits and local commits descended from the pinned base can be rebuilt. Setup rejects unrelated simulator revisions or a changed project patch; for those cases use a separate baseline checkout/work directory. Do not run two builds against the same work directory. If an initial fetch fails, finish or move the incomplete checkout yourself before retrying; setup never deletes it.

After the build job **finishes successfully**, submit the small verification job:

```bash
sbatch slurm/smoke_test.sbatch
```

Check `logs/build-<job-id>.log` and `logs/smoke-<job-id>.log`. The smoke job needs no original dataset. It runs the bundled litmus and checks for five completed requests, no suppressed packets, and observed invalidations/upgrades. A successful check prints `Litmus PASSED`. It is a setup/coherence sanity check, not a data-value correctness test.

Jobs should be submitted from this repository root. Slurm opens log files before scripts run, so create `logs/` first. The scripts retain the original class account and `standard` partition; override them with `sbatch --account=... --partition=...` if needed. To submit from another directory, use `sbatch --chdir=/absolute/path/to/this/repo /absolute/path/to/this/repo/slurm/smoke_test.sbatch`.

`source env.sh` makes the same paths available interactively. Configurable paths include `CXL_WORK_ROOT`, `SIMCXL_DIR`, `CXL_VENV`, `GEM5`, `RESULT_ROOT`, `OUTPUT_DIR`, and `OUTPUT`. Use absolute paths without whitespace for replay inputs and outputs. Modules can be changed using the four `CXL_*_MODULE` settings in `env.example.sh`.

Run the shared and sharded experiments with:

```bash
export TRACE="$PWD/traces/two-accel/shared-accel0.csv"
export TRACE2="$PWD/traces/two-accel/shared-accel1.csv"
export NAME=dlrm-shared
sbatch slurm/run_cxl_trace.sbatch

export TRACE="$PWD/traces/two-accel/sharded-accel0.csv"
export TRACE2="$PWD/traces/two-accel/sharded-accel1.csv"
export NAME=dlrm-sharded
sbatch slurm/run_cxl_trace.sbatch
```

Results go into `$RESULT_ROOT/<name>-<job-id>/`. Existing output directories are not overwritten.

## Shared traces and dataset

One team member can prepare the shared inputs with:

```bash
sbatch slurm/prepare_shared_data.sbatch
```

This publishes the committed `trace-v2` and `two-accel` traces and downloads only the dataset variant used by this experiment into:

```text
/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shared_data/eecs570-cxl-coherence/
├── datasets/fbgemm_t856_bs65536_0.pt.gz
└── traces/{trace-v2,two-accel}/
```

The [official Meta dataset](https://github.com/facebookresearch/dlrm_datasets/blob/main/embedding_bag/2021/fbgemm_t856_bs65536_0.pt.gz) is synthetic embedding-lookup data with production-representative reuse patterns. `configs/dlrm-dataset.json` records its public URL, expected size (1,174,963,458 bytes), and Git LFS SHA256. The job verifies the download before making it available, grants the class group read access, and refuses to replace conflicting existing files. Re-running it verifies and reuses the inputs. For traces alone: `sbatch slurm/prepare_shared_data.sbatch --traces-only`.

`CXL_SHARED_ROOT` and `DATASET` default to these shared locations and can be overridden in `env.local.sh`. Parent shared directories must already be available to your account. Generated traces and simulation outputs still go into each person's work directory. Keep published trace versions fixed; choose a new directory/version when changing inputs. Scratch is temporary, not backed up, and subject to the [60-day inactivity purge](https://documentation.its.umich.edu/arc-hpc/greatlakes/policies); retain durable copies elsewhere.

To replay shared inputs, set `TRACE` and `TRACE2` to CSVs under `$CXL_SHARED_ROOT/traces/` after `source env.sh`.

## Optional dataset processing

After the shared-data job finishes successfully, submit:

```bash
sbatch slurm/inspect_dlrm.sbatch
sbatch slurm/generate_dlrm_trace.sbatch
```

These jobs use `python/3.10.4` and `pytorch/2.0.1` from the cluster modules, independently of the build virtual environment. Submit from a shell without an activated virtual environment. The original dataset and PyTorch are unnecessary for trace replay. Dataset jobs read the shared input and write per-user outputs; they do not change the dataset. Agree on the input path and record its SHA256 alongside experiment notes if regenerating comparable traces.

## Simulator version

Base commit: `edddc2054bcdafdc7537b20c99605f2181bda9dc` (SimCXL `main`).

`patches/simcxl-csv-two-accel.patch` adds CSV replay and one private device L1 per trace generator. Setup applies it automatically. For background on upstream compiler and system dependencies, see the [SimCXL instructions at the pinned revision](https://github.com/TianheMICALab/SimCXL/blob/edddc2054bcdafdc7537b20c99605f2181bda9dc/README.md). The setup relies on cluster-provided system libraries and does not install OS packages.

## Traces

`traces/trace-v2/` is the representative single-accelerator sample. `traces/two-accel/` is the litmus and the shared/sharded pair. Each DLRM trace uses the first 256 rows (2048 lines of 64 bytes). Both accelerators read every line, accelerator 0 writes each line once, then both reread. The sharded accelerator 1 uses the same pattern shifted by 16 MiB. Phases start 40 million ticks apart.

Regenerate the pair from the row map:

```bash
python3 scripts/generate_two_accel_traces.py \
  --row-map traces/trace-v2/row_map.csv \
  --output-dir traces/two-accel
```

## Runs

| Run | Job | Stats |
|---|---|---|
| Litmus | 65710174 | `results/litmus-65710174/stats.txt` |
| Shared DLRM sample | 65710242 | `results/dlrm-shared-65710242/stats.txt` |
| Sharded DLRM sample | 65710243 | `results/dlrm-sharded-65710243/stats.txt` |

Shared minus sharded:

| Metric | Difference |
|---|---:|
| Simulated exit time | 0 |
| Invalidations | +2,048 |
| Upgrades to exclusive | +2,048 |
| GETX | 0 |
| Coherence message bytes | +491,520 |
| Average read latency | +37,887 ticks |

Exit time matches because both traces issue their last request at the same tick and then drain for 1 ms. The coherence cost is the invalidations, the upgrades, and the extra read latency on the accelerator that only reads.
