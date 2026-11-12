# Pooled sharded run

This is the production-style baseline on the eight-shard gem5 machine. The sample's accelerator does not read another accelerator's embedding rows. It sends the ids. The owner reads its own rows and sends back one 512-byte pooled result per bag. After every lookup in the slice, the sample's accelerator sends one 512-byte gradient to each owner that contributed. The owner then writes each distinct row once, locally.

The slice is 128 examples and 4 tables from `fbgemm_t856_bs65536_0.pt.gz`, split across eight accelerators. Requests are 500 ns apart. Retries were zero. All 381,379 memory operations completed. Job `65937061`. Output: `results/sharded-pooled-128x4-65937061/`.

## What the run did

| Movement | Count | Where it goes |
|---|---:|---|
| Ids sent to the owner | 20,651 | Across the switch |
| Owner lookups of its own rows | 23,547 rows, 188,376 lines | Local memory |
| Pooled results sent back | 2,360 | Across the switch |
| Gradients sent after the lookups | 2,360 | Across the switch |
| Owner row updates | 16,824 | Local memory |

The switch carried **2.6 MB**. That is exactly the ids, the pooled results, and the gradients. The owner's lookups and the final updates are not in that number.

Average read latency was **66 ns**. Those reads are the owner looking up its own memory, with no queue. Average write latency was **38 ns**. That average mixes the small id writes, the 512-byte results and gradients, and the local updates, so it is not the time for one gradient by itself.

The gradient phase happens once, after all lookups in this slice. It is one message per bag per remote owner, not one message per row. A second batch would do it again. This file is one batch.

## How this sits next to the other results

On this same slice, the earlier gem5 trace made each accelerator read the remote rows itself. That put **10.6 MB** on the switch and the average read was **142 ns**, because the row itself crossed the machine. Pooling cuts the switch traffic to **2.6 MB**. The row stays with the owner. Only the id, one result, and one gradient move.

On the full batch, the Python count of this same pattern is about **85 GB** of useful data in one step: 3.3 GB of ids, 40.7 GB of pooled results, and 40.7 GB of pooled gradients. The valid/invalid cache on that batch moves about **96 GB**, because it fetches the full rows and then invalidates 83.1 million copies. This gem5 run is the timed, small version of the 85 GB side. It does not include the invalidations.

The two-chip cache run remains the timed coherence number: about **1,900 extra bytes per shared row** and about **38 ns** on the next read. It did not replay these 128 examples.

## What not to claim

Do not report 66 ns or 38 ns as the cost of one embedding bag or one gradient. Do not report 2.6 MB as the full-batch cost. The 85 GB figure is that cost, and it is bytes, not time. Do not report a speedup against coherence. The eight-shard machine has no caches, and the two-chip machine did not run this trace. The transfers are ordinary memory writes of the right size, not an NCCL reduce-scatter.
