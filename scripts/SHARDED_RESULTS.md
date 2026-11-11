# Reading the sharded-memory results

On Great Lakes, run `bash scripts/run_sharded_memory.sh` from the repository
root to submit a job, wait for it, and print its summary in your terminal.
Alternatively, run `sbatch slurm/run_sharded_memory.sbatch` after creating
`logs/` to submit without waiting. After gem5 succeeds, the job prints a
summary into `logs/sharded-memory-JOBID.log` and saves the same summary in
`results/sharded-memory-JOBID/summary.txt`.

For an existing run:

```bash
python3 scripts/summarize_sharded_memory.py results/sharded-memory-JOBID
```

The summary reads the last complete statistics dump. A summary alone does not
establish successful Slurm job completion; check `sacct` and the log too.

| Metric | Interpretation |
| --- | --- |
| Simulated duration | Time on the modeled system's clock, including scheduled gaps and idle states. Not DLRM training time or embedding-phase completion time. |
| Simulator wall time | Real time used by the simulator, not target hardware performance. |
| Accelerator reads/writes | Responses received by each traffic generator, verified against this fork's `BaseTrafficGen::recvTimingResp`. |
| Average latency | Generator request-response latency, converted from ticks using `simFreq`. Averages combine local and remote accesses. |
| Memory-owner reads/writes | Requests accepted by each memory controller. Useful for checking routing and imbalance; not completion counters. |
| Fabric packets | Request and response traversals combined. One completed remote read normally produces two packets. |
| Fabric payload | Bytes carried as packet data across the central crossbar, excluding headers, link encoding, and protocol overhead. |
| Retries | Request retry notifications caused by backpressure; not packet loss. |
| Suppressed | Requests outside configured memory ranges. A nonzero value requires checking the workload address map. |

For the two-read routing test (job 65879789): generator 0 received two read
responses, memories 0 and 5 accepted one read each, the central fabric saw two
packet traversals and 64 payload bytes, and mean read latency was 104.233 ns.
The simulated duration was 1.001 ms, which includes a configured 1 ms idle
period. The mean does not tell us the local and remote read latencies separately.

The helper uses `numPackets` (in-range generated packets) minus read/write
responses to flag requests still awaiting responses. Zero does not prove that
every entry in a trace has been issued. Before larger distributed workloads,
replace the current global first-generator EXIT/fixed-window test approach
with explicit whole-workload completion tracking. Write responses also do not
prove that queued data has reached DRAM.

This configuration currently models memory and communication, with GPU compute
abstracted into trace scheduling. It has no cache-coherence traffic or cache
sharing statistics yet. Do not interpret payload divided by idle-inclusive
simulation duration as active-phase link utilization or training throughput.
