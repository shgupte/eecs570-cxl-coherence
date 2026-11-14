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
Done
- P1 traffic model (`step_traffic.py`): pool moves 1.4x fewer row transfers
  per step than deduplicated sharding, 7.7x fewer than per-lookup all-to-all.
- P2 placement (`sample_placement.py`): greedy sharing-aware placement cuts
  row fetches 29%; affinity placement also leaves 30% of fetches as rows the
  same GPU alone updated last. Caches: `dlrm_work/criteo/cache/{d1,pl_greedy,pl_aff2}`.
- P3 first pass (`gen_step_traces.py`, `summarize_step.py`): with the
  gradient path modeled, MESI with direct read-modify-write + affinity
  placement gives 40 µs/step vs 108 µs for the uncacheable baseline.
Running
- Sensitivity matrix (`run_step_matrix.sh`): bandwidth, latency, MLP, 8x scale.
- Stale-read accuracy runs (`staleness_train.py`, `slurm/v2/stale_train.sbatch`).
Open check
- MESI_Two_Level forwards data cache to cache without using the pool's
  port. CXL 3.0 back-invalidate routes it through the device. Quantify how
  much of the MESI gain depends on that (bw64 runs; if needed a protocol
  variant that forwards through the L2).

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
