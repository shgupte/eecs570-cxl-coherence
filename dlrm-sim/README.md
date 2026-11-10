# Trace-driven DLRM training simulator

This is a communication model of distributed DLRM embedding traffic. It does not
train a network, run gem5, or implement CXL. The only model in this tree is the
conventional sharded baseline: each embedding row has one owner, and every other
GPU fetches that row again on every step.

A later coherent-cache model can reuse the same batches, the same sample
partition, and the same owner-update schedule. That model is not implemented.

## What was on disk

Meta's `dlrm_datasets` layout, inspected from the local copy rather than assumed:

`/scratch/eecs570f26s001_class_root/eecs570f26s001_class/shgupte/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_0.pt.gz`

| Field | Shape | Dtype |
|---|---|---|
| `indices` | `(468084223,)` | float32 |
| `offsets` | `(56098817,)` | int64 |
| `lengths` | `(856, 65536)` | int32 |

The payload is a 3-tuple in that order. `56098817 = 856 * 65536 + 1`. Every
offset span matches `lengths`, so sample and bag boundaries are recoverable.
Bags are table-major: bag `table_id * 65536 + sample_id` is
`indices[offsets[bag] : offsets[bag + 1]]`. Empty bags are real (table 0,
sample 0 has length 0; the first stored row belongs to sample 1). Duplicate
row ids are kept.

Every stored index in this file is finite, integral, and non-negative. The
maximum is 12,543,668, which is below `2**24`, so the float32 encoding does
not round any row id in this file. The parser checks those conditions and
keeps the float32 array, casting each GPU window to int64 when it is read.

The other `fbgemm_t856_bs65536*.pt.gz` names in that directory are Git LFS
pointer files (135 bytes), not batches. `locality_stats.txt` describes the
un-suffixed `fbgemm_t856_bs65536.pt` object, which was not downloaded. This
simulator does not treat a replay of `_0` as a multi-step production trace.

Files whose `lengths` tensor is not 2-D are rejected. The loader does not
invent sample boundaries.

## Models

Eight GPUs by default. Samples are contiguous and disjoint. 65,536 samples
become eight ranges of 8,192. If the count does not divide evenly, the first
`batch_size % num_gpus` GPUs take one extra sample. Bags stay intact: sample
id, table id, and duplicate row ids are still there when communication is
counted. `(0, 10)` and `(1, 10)` are different embeddings.

The default model is `torchrec`. `--model row-transfer` keeps the earlier
reference. `--ownership modulo` keeps `row_id % num_gpus`. The default
ownership is contiguous, the TorchRec row-wise split: blocks of
`ceil(num_rows / num_gpus)`, with a short tail when the length does not divide
evenly. The access file has no table length, so each table is treated as
`max(observed row id) + 1` rows. Both models in one comparison must use that
same map. There is no remote cache in either model. Each distinct accessed
row is updated once, after gradients are aggregated.

### TorchRec pooled baseline

This follows `RwPooledEmbeddingSharding` as a byte count, not as a PyTorch run.

1. Every remote index in a bag, including duplicates, is an id request
   (`request_bytes`, default 8).
2. The owner looks up its own rows and builds one partial sum per nonempty
   `(sample, table, owner)`.
3. `reduce_scatter_bytes` approximates the collective TorchRec uses to combine
   those partials: `(num_gpus - 1) * samples * tables * embedding_bytes`.
   Empty bags are included, because the real collective is sized by batch and
   feature count. `logical_remote_partial_bytes` is the sparse count and is
   not that collective.
4. Backward all-gathers pooled gradients. It does not send one gradient per
   row. The owner then applies that pooled gradient to its rows. Those
   applications are `row_gradient_applications`, not network traffic.

`forward_bytes` is the id traffic plus the reduce-scatter approximation.
`backward_bytes` is the all-gather approximation.

### Row-transfer reference

`--model row-transfer` deduplicates rows inside a GPU, fetches one full vector
per unique remote row, and sends one gradient per unique remote row. That is
the model from the first simulator. It is the backward path a later
coherent cache should use if the requesting GPU pools locally, because that
GPU would hold rows rather than pooled outputs.

Defaults `embedding_dim=128` and `bytes_per_element=4` are configuration. The
access file does not store them. They match the 512-byte row used by the
earlier CXL traces. Sum and mean pooling move the same number of vectors.

### Valid/invalid cache

`--model coherent --protocol vi` keeps a remote row cache on each GPU.
Readers fetch full rows and pool locally. `--cache-bytes` sets the capacity.
Omit it, or pass `unlimited`, for a cache that holds every fetched row.
An update invalidates every copy of a row that was read, so the next step
misses on those rows. Duplicate touches in the same step can hit. Gradients
are one vector per remote row. Invalidation and acknowledgement sizes default
to 64 bytes each.

## Run

On the course cluster:

```bash
module load python/3.10.4 pytorch/2.0.1
cd ~/dlrm-sim
python -m pytest -q
python main.py --demo --output-dir results/demo
python main.py --demo --model row-transfer --ownership modulo --output-dir results/demo-rows
python main.py \
  --dataset /scratch/eecs570f26s001_class_root/eecs570f26s001_class/shgupte/dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536_0.pt.gz \
  --output-dir results/fbgemm-torchrec
```

`--samples` and `--tables` take a contiguous slice and record the run as
`meta-batch-subset`. `--repeats` replays one batch and labels it
`synthetic-repeat`. That is not measured cross-step locality.

Outputs:

* `structure.json` — tables, samples, raw accesses, parser notes
* `metrics.json` — per-step metrics, pair matrices, assumptions
* `steps.csv`, `pairs.csv`

`cache_hits` is always 0. A coherent model should keep `collect_unique_accesses`
and `apply_owner_updates`, and it should send per-row gradients if pooling
moves to the requesting GPU.

## Tests

The tests are hand-built batches, not the Meta file.

* Every sample is on one GPU, and its bag lookups stay on that GPU.
* Contiguous shards match TorchRec's block sizes, including a short table.
* One bag can name several owners. Two rows on the same owner are one partial.
* Duplicate ids are separate id requests and one partial. Two samples can share a row.
* Pooled backward traffic is not one gradient per row.
* Row-transfer and TorchRec, given the same ownership, update the same rows once.
