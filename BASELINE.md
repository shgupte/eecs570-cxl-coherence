# Baseline for the milestone

The question is whether GPUs should keep their own copies of embedding rows, or ask the owner each time. The baseline is the second choice. Each row has one owner. Another GPU that needs it reads the owner's memory. The owner applies the update. There is no cached copy and no invalidation.

Three measurements support that baseline. They are not one experiment, and none of them is a full training-step speedup.

## 1. How much data moves on the full batch

Source: Meta file `fbgemm_t856_bs65536_0.pt.gz`, one batch of 65,536 examples and 856 tables. Counted in `dlrm-sim` on `main`, not timed in gem5. Eight GPUs, contiguous example ranges, contiguous row owners. An embedding row is 512 bytes.

Useful sharded traffic for one step is about **84.6 GB**:

- 3.3 GB of row ids sent to the owners
- 40.7 GB of pooled results sent back
- 40.7 GB of pooled gradients

A second figure, about 201 GB each way, counts empty slots in the software collective. That is not embedding data. The comparison below uses 84.6 GB.

The same batch with an unlimited valid/invalid cache moves about **96.4 GB** per step. Readers fetch full rows and pool locally. Every row that was read is updated, so every fetched copy is invalidated before the next step. The second pass costs the same 96.4 GB. The extra 11.8 GB is about 10.6 GB of invalidations and replies, plus a bit more because full rows are sent instead of one pooled result. The cache does absorb 326 million repeated lookups inside the step. It still misses 83.1 million times, once per distinct remote row.

61.6 million rows are updated. A typical row is used by 1.5 GPUs. Only about 1 in 80 rows is used by all eight. Most of the coherence traffic is one remote copy on each of tens of millions of rows, not many GPUs sharing each row.

## 2. What an invalidation costs on two chips

Source: the earlier SimCXL runs, two accelerators with private caches. Both chips read the same 256 rows (2,048 lines of 64 bytes), one chip writes, both read again. The other run gives the second chip different rows.

Shared minus disjoint:

- 2,048 extra invalidations
- 491,520 extra coherence bytes, about **1,920 bytes per row**
- about **38 ns** extra on the next read

The simulated run does not finish later, because both traces were built to end together and then wait. On a larger 64-example slice the 128 KB cache could not hold the rows, so those misses are a full cache, not sharing.

Asking the owner for one row, in the byte model, is about **1,024 bytes**: 512 out and 512 back, and no invalidation. On that cold shared pattern, keeping the copy cost more message bytes than sending the value once.

## 3. How long a remote read takes on eight shards

Source: `configs/sharded_memory.py` on `baseline-setup`, job `65933171`. Eight accelerators, each with 2 GB of DDR5 and no caches. A remote read crosses a central switch (20 ns onto the switch and 20 ns off the other side) and then hits the owner's DRAM.

The trace is a real slice of the Meta file: **128 examples and 4 tables**, not the full batch. Examples are split across the eight accelerators. Each row lookup is eight 64-byte reads of a 512-byte row, at a packed address in the owner's shard. Raw row ids do not fit in 2 GB. After the reads, the owner writes each distinct row it owns once. That write is local. Other chips see it only if they read that address later.

Requests were issued 500 ns apart so they would not queue. Retries were zero. All 322,968 requests completed.

- Average read latency: **142 ns**. About 88% of the row reads were remote, so this is mostly the time to read another accelerator's memory. Local reads are included in the average and make it a little faster than a pure remote read.
- Average write latency: **12 ns**. This is the acknowledgment that the write was accepted, not the time until the new value is stored.
- Data on the switch: **10.6 MB**, matching the remote reads only.

An earlier 16-example run issued a line every nanosecond and averaged 252 ns, with 15,211 retries. That number is the queue, not the memory path. Use 142 ns, not 252 ns.

## What this baseline is for

Use 142 ns as the time to fetch a remote line from the owner on this eight-shard machine, for that slice. Use 1,024 bytes and 84.6 GB as the data the normal path moves when the owner sends a pooled result. Use 1,920 bytes and 38 ns as the extra cost of one other cached copy on the two-chip machine, and 96.4 GB as the full-batch cost of fetching rows and then invalidating them every step.

Do not report a single speedup. The eight-shard run has no caches. The two-chip run did not replay these same lookups. The Python counts have no clock.
