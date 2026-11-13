# Coherence baselines for CXL-pooled DLRM training (v2)

Question: when GPUs read shared embedding rows from a CXL pool, does hardware
coherence (MESI) beat simpler no-coherence designs?

**Answer so far (2 GPUs + owner, 1,024 hot Criteo rows, 256-cycle CXL hop):
no.** Software flushing at the write boundary ties or beats MESI at every
write interval, and both beat an uncacheable pool only when writes are
batched across steps. See [Results](#results).

## What is simulated

```
accel0 (GPU0)   accel1 (GPU1)   owner (other owners)   host CPU (idle)
  L1 1MiB         L1 1MiB         L1 1MiB               L1
     \______________|________________|____________________/
                    |   point-to-point network, 6 virtual nets
          shared L2 8MiB 16-way, inclusive  <- CXL hop charged here
                    |                          (l2 latency, 256 cycles each way)
          directory (MESI home)
                    |
          DDR5-4400 3GiB  (the pooled memory)
```
SimCXL (gem5, Ruby, `CXL_MESI_Two_Level`), 2.4 GHz, 64 B lines. Devices are
CSV trace generators (≤32 outstanding requests), not real GPUs.
**Not modeled:** multiple hosts/coherence domains, CXL switch and link
bandwidth, GPU compute, data values (so no accuracy effect of staleness), more
than 2 reader GPUs.

## Simulator changes (`patches/simcxl-v2-barrier-l2lat.patch`)
Apply to SimCXL `edddc2054b`. It already contains `simcxl-csv-two-accel.patch`.

- **Barrier op in CSV traces** `b,<id>,0,<tick>`: the generator waits until its
  own requests finish and every generator reaches barrier `<id>`. Release
  ticks are logged (`barrier N release tick T`) and give phase/step times.
- `CXLMESITwoLevelCacheHierarchy(l2_latency=256)`: the CXL hop.
- `configs/cxl-trace.py` flags (defaults reproduce the original team runs):

| Flag | Default | Meaning |
|---|---|---|
| `--l1d-size` | 128KiB | per-device L1 (we use 1MiB) |
| `--l2-size` | 512KiB | shared L2 (we use 8MiB) |
| `--l2-latency-cycles` | 256 | CXL hop, each direction |
| `--cxl-extra-latency-ns` | 0 | extra memory-controller latency |
| `--max-outstanding` | 32 | per-generator requests in flight |

## Trace generator (`gen_multistep_traces.py`)
Each step = 3 phases separated by barriers: **read** (each GPU reads all 8
lines of each hot row it used; the forward-pass critical path), **write**
(each touched row written once by its owner), **push** (unused now). Modes
change only addresses, so request counts and order are identical.

| Flag | Meaning |
|---|---|
| `--hot` | `hot_rows.npz` from `reuse_analysis.py` (per-step GPU masks, owners) |
| `--rows 1024 --steps 16` | hot rows / steps replayed |
| `--mode shared` | **MESI**: one copy of each row; L1s cache it, writes invalidate |
| `--mode uncached` | **Uncacheable pool** (CLAY-style): every read misses L1 every step; no coherence messages |
| `--mode swflush` | **Software-managed caching**: cache freely, drop everything at each write boundary |
| `--stale k` | owner writes every k steps (batched updates / gradient accumulation) |
| `--alias 4` | uncached/swflush: address aliases cycle every 4 steps/windows so misses can hit L2 |
| `--warm` | uncached/swflush: warm-up step loading all aliases into L2 (fairness: MESI misses hit L2) |
| `--mode sharded`, `--mode push` | **legacy, not valid baselines**: readers never see updates |

Emulation biases: swflush does not pay the flush/writeback; uncached writes
fetch the line before writing (only affects the write phase).

## Replicating

1. **Build the simulator**: apply the patch to SimCXL `edddc2054b`, then
   `SIMCXL=<checkout> sbatch slurm/v2/build_baseline.sbatch`.
2. **Data**: Criteo 1TB parquet (HF `criteo/CriteoClickLogs`), then
   `criteo_to_batches.py` → `reuse_analysis.py --row-shift 26` produces
   `hot_rows.npz` (see `slurm/v2/criteo_reuse.sbatch`).
3. **Sweep** (from the repo root):
   ```
   GEM5=<checkout>/build/X86_CXL_MESI/gem5.opt RESULT_ROOT=<dir> \
   HOT=<dir>/hot_rows.npz LATS="256" KS="1 2 4 8" analysis/run_coherence_sweep.sh
   ```
   12 jobs (3 modes × 4 write intervals), ~1 h each.
4. **Summarize**: `python3 analysis/summarize_sweep.py <RESULT_ROOT> sel logs`
   — read/step µs averaged over steady-state steps 4–15, plus invalidations,
   GETX, L1 hits/misses and DRAM reads.

## Results
`results_gem5_coherence_baselines.csv`. Read phase / whole step, µs, 256-cycle hop:

| write every k | MESI | software flush | uncacheable |
|---|---|---|---|
| 1 | 42.3 / 70.1 | **32.8 / 55.3** | 32.8 / 55.3 |
| 2 | 25.2 / 39.1 | **20.4 / 32.1** | 32.8 / 35.9 |
| 4 | 16.6 / 23.5 | **14.2 / 20.5** | 32.7 / 34.3 |
| 8 | 10.8 / 15.5 | **10.0 / 14.3** | 32.7 / 33.7 |

MESI at k=1 vs hop 128/256/512: read 32.6 / 42.3 / 64.5 µs.

Why MESI loses: in synchronous DLRM training writes happen at a known point
(the barrier), every row read in a window is written at its end, and sharing
is coarse-grained. Precise hardware invalidation therefore does the same work
as a software flush, but per line with messages, and the re-read after an
update takes an extra hop through the owner's dirty copy.

Other scripts: `criteo_to_batches.py`, `reuse_analysis.py` (Criteo: 35% of
rows recur next step; top-1K rows = 69% of lookups), `compare_batches.py`
(Meta `dlrm_datasets` batches are synthetic near-copies, not usable for
reuse), `selective_model.py` (idealized byte model, `results_selective_model_criteo_d1.csv`).

## Next
Charge the flush cost in swflush; 8 reader GPUs; asynchronous updates (no
barrier between write and next read), the one regime where hardware
coherence should have an advantage.

## 8-GPU validation matrix (`gen_ngpu_traces.py`, `run_ngpu_matrix.sh`)
8 GPUs, Criteo masks, 5 modes (`mesi`, `uncached`, `swflush`, `swinv` =
software selective invalidation, `update` = ideal write-update/push), swept
over write interval k, write fraction p, link bandwidth, hop latency, MLP,
hot-set size, batch window and GPU count. Results:
`results_gem5_ngpu_matrix.csv` (`summarize_ngpu.py`). Step time µs (best in bold):

| regime | MESI | swflush | swinv | uncached | update |
|---|---|---|---|---|---|
| k=1 base | 50.8 | 46.0 | 46.9 | **38.9** | 137 |
| k=4 base | 18.8 | **17.5** | 17.7 | 34.6 | 40.5 |
| k=1 p=0.1 (read-mostly) | **9.2** | 34.6 | 10.1 | 38.9 | 17.8 |
| k=4 p=0.1 | **8.3** | 14.8 | 8.6 | 34.6 | 9.8 |
| k=4 bw64 | **15.5** | 16.9 | 17.1 | 33.6 | 17.2 |
| k=4 hop 512 | **25.1** | 25.8 | 26.2 | 61.1 | 40.7 |
| k=4, 2 GPUs | **18.6** | 23.2 | 23.4 | 35.0 | 13.4* |
| k=4, 4 GPUs | **18.9** | 19.3 | 19.5 | 34.5 | 19.9 |

*update wins with 2 GPUs (only one holder to push to).
MESI wins where updates are rare (p ≤ 0.25), bandwidth is ample, or few GPUs
share; it loses at 8 GPUs with frequent writes and on a constrained pool port.
Every MESI win comes from the software modes' publish (sync) phase; read
phases are within ~3%. Software sync is overcharged here (RFO + write-back per
line), so the MESI wins are upper bounds. Net: **MESI is within ±10% of the
best software scheme almost everywhere, not uniformly worse.**
