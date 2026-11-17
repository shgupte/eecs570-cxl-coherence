---
title: "Coherent caching for a CXL embedding pool in multi-GPU DLRM training"
subtitle: "EECS 570 project summary"
date: "2026-11-17"
geometry: margin=1in
fontsize: 10pt
---

## Problem

Recommendation models (DLRM) keep most of their parameters in embedding
tables that reach hundreds of GB to several TB. Every training step, each
GPU reads the rows its samples use and then updates them. The tables do not
fit in GPU memory at production scale, and buying GPUs for their memory is
expensive (about \$390 per GB against about \$40 per GB for server DRAM).

A CXL memory pool shared by the GPUs solves the capacity problem. The
question we study is how the GPUs should use that pool during training,
when several GPUs read and update the same rows in the same step.

## Motivation

CLAY (ICS 2024) places embedding tables in CXL memory for inference and
avoids coherence by treating the memory as uncacheable. Extended to
training, that means every row is fetched from the pool and written back
every step, even when only one GPU ever uses it. Two observations suggest
this is wasteful:

- **Most rows are private.** In a step of 65,536 Criteo samples on 8 GPUs,
  about 80% of the rows touched are used by a single GPU.
- **CXL has no cache-to-cache forwarding.** Standard MESI lets GPUs cache
  rows, but every write to a shared row must pass ownership through the
  pool, which is slow.

## Proposed solution

Three changes to the uncacheable pool:

1. **MESI coherence between the GPUs and the pool.** A GPU that is the only
   user of a row keeps it cached and updates it locally, with no traffic.
2. **Merged gradient updates (deltas).** For a shared row, each GPU sends
   its gradient as a delta and the pool adds it in. No GPU takes exclusive
   ownership and no copies are invalidated. This works because gradient
   updates are commutative.
3. **Sample placement.** The data loader assigns samples to GPUs so that
   samples using the same rows land on the same GPU, which makes more rows
   private. Training is unchanged because the summed gradient does not
   depend on which GPU computes which sample.

We implemented the protocol in gem5 (SimCXL, Ruby), including new request
types for deltas and for pool-mediated downgrades, and found and fixed a
stale-copy bug in the downgrade path along the way.

## Results

All experiments use Criteo 1TB batches, 8 GPUs, 512-byte rows. gem5 replays
a 1/64 sample of the rows; a link-level model computes full-batch traffic
with real link speeds (63 GB/s per CXL port).

**Each change helps (gem5, pool with 8 ports, time per step):**

| Design | Step time | Speedup |
|---|---|---|
| Uncacheable pool (CLAY-style) | 75.8 µs | 1.0x |
| + sample placement | 41.4 µs | 1.8x |
| + MESI through the pool | 29.2 µs | 2.6x |
| + merged gradient updates | 23.7 µs | 3.2x |

**Against other pooled designs (full batch, 8 ports, embedding traffic per step):**

| Pooled design | One ID per feature | Multi-hot (MLPerf recipe) |
|---|---|---|
| Uncacheable (CLAY-style) | 0.67 ms | 15.7 ms |
| Ours | 0.38 ms | 6.8 to 12.6 ms |
| Near-memory pooling (approach of ReCXL, RecDM) | 0.81 to 4.35 ms | 0.81 to 4.35 ms |
| Near-memory pooling + our caching | not run | 0.57 to 4.15 ms |

- **Against the uncacheable pool:** 2.2 to 3.2 times faster with one ID per
  feature, and 1.2 to 2.3 times faster on multi-hot data.
- **Against near-memory pooling:** about 2 times faster with one ID per
  feature, where a cached private row needs no transfer. With large bags the
  near-memory pool is faster, because it sends one vector per bag. The two
  compose: adding our caching to a near-memory pool is 5 to 13% faster than
  pooling alone with ideal placement.
- **Pool ports:** the pool needs about 8 ports. With one port our design
  takes 65.0 µs in gem5 against 23.7 µs with 8.
- **Placement is safe and practical:** after 40 steps of real training the
  weights differ by 1.8e-14 and AUC is identical. Run as a pipeline it keeps
  about two thirds of its benefit.
- **GPU cache:** about 100 MB per GPU is enough with one ID per feature.
- **Robustness:** results repeat on a second step window, a second Criteo
  day and an 8x larger trace.

## Scope and limits

- **Not competitive with sharding over NVLink.** When tables fit in GPU
  memory, sharded all-to-all is 3.3 times faster on embedding traffic. Our
  design is for tables that do not fit.
- **Host memory with the same scheduling ties our pool** on step time; the
  pool's advantage there is freeing the CPU cores that move rows.
- **A software schedule comes within 18% of the hardware protocol,** because
  the data loader knows future batches.
- **The near-memory comparison is our model of that approach,** not a
  reproduction of ReCXL or RecDM, and it ignores the pool's internal memory
  bandwidth. The multi-hot data is synthetic.
- **No GPU has a CXL port today,** and the protocol was tested on our traces,
  not formally verified.

Full report with figures and all experiments: `analysis/report/REPORT.pdf`.
