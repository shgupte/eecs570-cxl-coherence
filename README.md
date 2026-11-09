# Two-accelerator CXL coherence experiment

SimCXL comparison of two accelerators reading and updating the same DLRM embedding lines versus the same requests on disjoint lines.

The simulator checkout stays at `~/cxl_gl_agent/sources/SimCXL`. This repo holds the patch, the replay scripts, the traces, and the stats from the runs that count. It does not contain the gem5 build, the guest images, or the 1.09 GiB dataset.

## Simulator change

Base commit: `edddc2054bcdafdc7537b20c99605f2181bda9dc` (SimCXL `main`).

`patches/simcxl-csv-two-accel.patch` adds a CSV traffic-generator mode and one private device L1 per trace generator. Apply it from a clean checkout of that commit:

```bash
cd ~/cxl_gl_agent/sources/SimCXL
git checkout edddc2054bcdafdc7537b20c99605f2181bda9dc
git apply /path/to/eecs570-cxl-coherence/patches/simcxl-csv-two-accel.patch
```

Then rebuild `X86_CXL_MESI/gem5.opt` with `slurm/build_simcxl.sbatch` from `~/cxl_gl_agent`. The Slurm scripts in this repo are copies of the ones that run there. They expect `sources/SimCXL` and `configs/cxl-trace.py` under the submit directory.

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
