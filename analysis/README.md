# v2 analysis and simulator changes

## Simulator (`patches/simcxl-v2-barrier-l2lat.patch`)
Full patch against SimCXL `edddc2054b`; it **includes** `simcxl-csv-two-accel.patch`, so apply it instead of that one.
- CSV barrier op `b,<id>,0,<tick>`: a generator waits for its own outstanding requests and for every CsvTraceGen to reach barrier `<id>`; later ticks are relative to the release. Release ticks are printed (`info: ... barrier N release tick T`) and give phase/step times.
- `CXLMESITwoLevelCacheHierarchy(l2_latency=256)`: SimCXL models the CXL hop as the L2 request/response/to-L1 latencies. Default unchanged.
- `configs/cxl-trace.py`: `--l1d-size`, `--l2-size`, `--l2-latency-cycles`, `--cxl-extra-latency-ns` (memory-side). Defaults reproduce the original runs exactly (litmus/shared/sharded stats identical).

## Scripts
- `reuse_analysis.py` — per-batch row/GPU sets, cross-step reuse, hot-set persistence, hot rows for gem5.
- `compare_batches.py` — shows Meta `dlrm_datasets` batches are near-copies (99.5% same rows, counts corr 0.9998): independent synthetic samples, not consecutive steps.
- `criteo_to_batches.py` — Criteo 1TB parquet (HF `criteo/CriteoClickLogs`) to Meta-style batches; 26 tables, row = hash % 40M.
- `gen_multistep_traces.py` — multi-step hot-row traces (accel0=GPU0, accel1=GPU1, owner=other owners) with barriers; `--mode sharded` puts every generator in a private region.
- `summarize_sweep.py <results> <prefix> <logdir>` — stats + step times, shared minus sharded.

## Results so far (12 Nov 2026)
Criteo day 1, 32 consecutive batches: 35% of rows recur next step; top-1K rows persist 84% and take 69% of lookups (top-10K: 84%). Meta synthetic: 99.5% recur (artifact).
gem5, 1024 Meta hot rows, 16 steps, 1 MiB L1: sharded 13.8 us/step; VI-shared 48.5/68.9/109.9 us/step at L2 hop 128/256/512 cycles; ~4.2 KB extra traffic per shared row per step vs a 1 KB send. Sharded does not yet charge the send.
