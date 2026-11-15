# agent.md — autonomous campaign on CXL-pooled DLRM training

Written 2026-11-14. The user is away for the day. Work autonomously; do not
stop to ask. Plan and rationale: `analysis/CAMPAIGN.md`.

## Mission
EECS 570 (parallel computer architecture) project. System: 8 GPUs sharing a
CXL memory pool that holds DLRM embedding tables, synchronous training,
Criteo 1TB. Baseline: CLAY (ICS'24) extended to training = uncacheable pool.
Find a design (coherence, consistency, scheduling, or the pool architecture
itself) with a real, defensible gain over that baseline, or bound the gain
honestly. Evidence for and against MESI must both be reported.

## Standing rules
- **Never stop at a dead end.** If a direction fails its gate, write down why
  (one paragraph in `analysis/CAMPAIGN.md`), pick the next direction from the
  list below, and continue. Stop only when the report is written and pushed.
- Check every positive result for simulator artifacts before believing it
  (earlier artifacts: cold aliases going to DRAM, clean-E write-backs,
  double-charged stores, the horizon hiding write-backs).
- Site policy: no recursive scans of /scratch, /nfs, /home roots; never poll
  squeue in a loop; wait with `srun --dependency=afterany:<ids> true`.
- Slurm: account `eecs570f26s001_class`, partition `standard`, only 2 jobs
  run at once. A gem5 run of a 1/64 trace takes about a minute.
- Python: `/home/bany/bin/python3`; pyarrow needs
  `PYTHONPATH=/scratch/engin_root/engin1/bany/pylib`.
- Commit to branch `v2-barrier-criteo-selective`; push when a phase is done.

## State (update this section as work proceeds)
Done (see `analysis/REPORT_2026-11-14.md`, second version)
- Whole-step traffic model, placement, staleness runs (first version).
- Explicit port topology (`configs/cxl-trace.py --topology pool|sharded`),
  sharded all-to-all baseline (`gen_shard_traces.py`), real protocol support
  for commutative updates and pool-mediated reads
  (`patches/simcxl-v4-commutative.patch`: `--commutative`,
  `--pool-mediated-gets`), port matrix of 112 runs (`run_port_matrix.sh`).
- Placement checks: `placement_check.py` (identical training), `placer.c`.
- Answer so far: per-lookup all-to-all 202 µs; deduplicated 45; with
  placement 34. Pool with 1 port 61-64 (loses); with 4+ ports 23-26 (wins
  1.3-1.44x) because private rows stay at their GPU. MESI + commutative
  updates equals MESI with forwarding without needing forwarding.
- Artifacts found this round: SimpleNetwork throttles only switch outputs;
  lazy write-back emulation undercharged gradient return; private-alias
  lines surviving in a large L1 (uncacheable designs now use a 64 KiB L1,
  ld_hit must be 0); warm-up engines must not send commutative updates.
Open
- 1/8-sample scale check of the port matrix (`X=8 run_port_matrix.sh`).
- Real step time to judge the placer cost; another Criteo day; multi-hot.
- Different link speed/latency for NVLink vs CXL beyond the sensitivity rows.

## Next directions if the current one dead-ends (in order)
1. Pool-mediated forwarding variant of the Ruby protocol, to remove the
   forwarding caveat.
2. A real commutative-update state (COUP-style) in the Ruby protocol.
3. Bounded-staleness consistency: which rows may be read one step stale
   (accuracy from `staleness_train.py`), turning dependent reads into
   prefetches.
4. Pool topology: several pool ports / multi-headed devices, tables placed
   to balance port load.
5. Hybrid: hot rows replicated in GPU memory with all-reduce, tail in the pool.

## Paths
- Repo `~/eecs570-cxl-coherence`; simulator `/scratch/engin_root/engin1/bany/simcxl-baseline`.
- Data and results under `/scratch/engin_root/engin1/bany/dlrm_work/`
  (`criteo/batches/d1`, `criteo/cache/*`, `traces/step`, `sim-results`, `stale`).
- Final report: `analysis/REPORT_2026-11-14.md`.
