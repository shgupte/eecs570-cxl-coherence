# Progress: DLRM coherence on SimCXL

EECS 570. Question: what coherence costs when embedding rows are updated through shared CXL memory instead of being owned and exchanged over the network. A negative or limited result is acceptable. Do not run full DLRM training or a 64-node cycle-level sim.

## Where things are

- Simulator checkout: `~/cxl_gl_agent/sources/SimCXL`, base `edddc2054bcdafdc7537b20c99605f2181bda9dc`. Binary: `build/X86_CXL_MESI/gem5.opt` (rebuilt 8 Nov 2026). Protobuf is off, so replay is CSV, not gem5 packet traces.
- Experiment repo: `/home/shgupte/eecs570-cxl-coherence`, GitHub `https://github.com/shgupte/eecs570-cxl-coherence`. The first commit has the SimCXL patch, the litmus/shared/sharded stats, and the two-accelerator traces. `traces/dp-split/` is the 64-example split. `dlrm-sim/` is the Python traffic counter. Its one-step result is `results/dlrm-torchrec-one-step/`.
- Dataset, not in git: `/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shgupte/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_0.pt.gz`. One batch. 65,536 examples, 856 tables, 468,084,223 lookups. The other `.pt.gz` names in that directory are Git LFS pointers, not batches.
- Slurm account `eecs570f26s001_class`, partition `standard`. Replay script: `~/cxl_gl_agent/slurm/run_cxl_trace.sbatch` with `TRACE` and optional `TRACE2`.

## What a sample is

One training example in the mini-batch. Each example has a bag of row ids per table (`lengths[table, sample]`). Those ids are looked up and pooled. The file has no dense features, labels, or embedding values. About 8 ids per table per example.

## Simulator setup

Two accelerators. Each has a private L1 (128 KiB, 8-way) and a CSV traffic generator. Both L1s share one L2 (512 KiB) and directory. Embeddings sit in 3 GiB host DDR5. There is no device memory. An idle CPU generator is attached and issues nothing. The patch is `patches/simcxl-csv-two-accel.patch`.

## What production training does

Large embedding tables are model-parallel. Each row has an owner. Other devices do not load that address. The owner reads the row, all-to-all sends the vector, receives the gradient, and writes the row once (fused optimizer, one update per unique row per step). Small tables are sometimes replicated, with an all-reduce and a local write on every replica. A directory home is not a software owner: the directory still lets every device cache the line and invalidates those copies on a write.

## Experiment 1: same lines vs disjoint lines

Both accelerators read 2,048 lines, accelerator 0 writes each once, both reread. The sharded run shifts accelerator 1 by 16 MiB. Those lines are the first 256 compact rows of table 18 (reuse about 1.07), not a hot set.

Litmus job 65710174 passed: one invalidation, and accelerator 1’s reread missed. Shared job 65710242 minus sharded job 65710243:

| Metric | Shared − sharded |
|---|---:|
| Simulated exit time | 0 |
| Invalidations | +2,048 |
| Upgrades | +2,048 |
| GETX | 0 |
| Coherence message bytes | +491,520 |
| Average read latency | +37,887 ticks |

Exit time matches because both traces end at the same issue tick and then drain 1 ms. This is the cost of one other cache holding a line. It is not a 64-node link result.

## Experiment 2: batch split across two accelerators

Regenerated from the same 64 examples and 12 tables. Accelerator 0 gets examples 0–31, accelerator 1 gets 32–63. Each reads its own examples in order. Accelerator 0 then writes every unique row either side touched, once. Both reread the rows they touched. Splitting examples is a fair stand-in for data parallelism. Splitting individual reads is not.

| Rows | Count |
|---|---:|
| Only accelerator 0 | 1,899 |
| Only accelerator 1 | 1,318 |
| In both halves | 304 |
| Writes | 3,521 |

Traces: `traces/dp-split/accel0.csv` and `accel1.csv` (also under `dlrm_work/dp-split` on scratch). Phase ticks were widened before the run so the write starts at tick 200,000,000 and the reread at 400,000,000. Generator: `~/cxl_gl_agent/scripts/generate_dp_traces.py`.

Replay job 65713803 completed. Both generators finished every request (accelerator 0: 40,480 reads and 28,168 writes; accelerator 1: 28,336 reads). Simulated time 1,545,948,911 ticks. Average read latency about 346,000 ticks on both. Invalidations 7,873, of which 5,757 were in accelerator 1’s L1. GETX 28,125. L1 replacements 88,368.

The working set does not fit. Accelerator 1’s rows are about 811 KiB and each L1 is 128 KiB, so most lines are evicted before the write. Accelerator 0 also writes rows that only accelerator 1 read, so many invalidations are the writer taking a line the other cache held alone.

## Conventional traffic count

`dlrm-sim/` counts messages for one training step. It does not train a model and it does not measure time. Eight GPUs each get 8,192 consecutive examples. Rows are split into contiguous blocks, the TorchRec row-wise rule. A table's length is not in the file, so it is taken as one past the largest row id seen. An embedding is 512 bytes (128 × 4), the same size used in the gem5 traces. Owners pool each bag and return one partial per nonempty sample, table, and owner. Gradients of those pooled outputs go back. Each distinct row is updated once.

On the full batch, 9 Nov 2026:

| Quantity | Value |
|---|---:|
| Rows updated | 61,644,628 |
| GPUs per row, average | 1.54 |
| Rows used by all 8 GPUs | 789,365 |
| Remote row ids | 409,564,021 (3.3 GB) |
| Remote partials, each way | 79,403,173 (40.7 GB) |
| Dense collective, each way | 201 GB |

The 40.7 GB is the useful embedding data. The 201 GB count includes an empty slot for every example and every table. Use 40.7 GB, not 201 GB, when comparing with coherence. One remote reader of one row is 512 bytes out and 512 bytes back, 1,024 bytes, with no invalidation. There is no second real batch, so this does not say whether the same rows are hot on the next step.

## What is established

If two accelerators cache the same embedding line and one writes it, the other copy is invalidated and the next read misses. On the cold 2,048-line sample that cost 2,048 invalidations, 2,048 upgrades, 491,520 extra coherence bytes, and about 38 ns of read latency. The batch-split run shows the same mechanism on real example lookups, and also shows that a 128 KiB L1 turns most of this slice into capacity misses.

The full-batch count and the gem5 runs are different experiments. Do not subtract 40.7 GB from the gem5 totals. The fair comparison on the cold 2,048-line sample is per row: coherence cost 1,920 extra bytes (491,520 / 256 rows) and about 38 ns on the next read; the ordinary send is 1,024 bytes and no invalidation. On that cold pattern, keeping one other copy cost more message bytes than sending the value once. Most rows in the full batch have one user, so they would not pay the coherence cost.

## What is not established

This does not show a CXL or PCIe link filling up, and it does not time a training step. The Python counter has no clock. Gem5 has not replayed the hot rows across several steps with a cache large enough to hold them. Only those rows, repeated, can say whether a shared copy beats the 1,024-byte send. There is one real batch, so a repeated trace is a reuse test, not the next measured production step. A 64-node claim stays a bandwidth estimate: extra bytes per sharer, divided by one link's bandwidth.

## Next step

Keep both tools. Use the Python count as the ordinary-send price. In SimCXL, replay a small set of rows that several GPUs touch, for several steps: read, one owner write, read again. Keep a same-sized set of single-user rows as the control. Make the L1 large enough, or the set small enough, that the rows fit. Compare the extra coherence bytes and the extra read delay with 1,024 bytes per other reader. Add a third accelerator only after that trace is stable.
