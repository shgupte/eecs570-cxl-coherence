# One-day campaign: what a CXL pool needs for DLRM training (2026-11-14)

Baseline: CLAY (ICS'24) extended to training — GPUs read embedding rows from a
CXL pool uncached. System: 8 GPUs, one pool, synchronous training, Criteo 1TB.

## Why the direction changes
1. The earlier gem5 slice (top-1K hot rows) is 73% of lookups but ~3% of the
   deduplicated row fetches. A step fetches ~300K (GPU,row) pairs over ~160K
   unique rows; 39% of pairs are rows touched by one GPU; 33% of rows recur
   next step.
2. The gradient return path (combining 8 GPUs' gradients for a shared row)
   was never modeled. It is where coherence designs differ.
3. In exact synchronous training the bytes are nearly conserved, so the levers
   are: who touches what (scheduling), what is on the critical path (timing),
   where gradients are combined (coherence), and how fresh reads must be
   (consistency).

## Question
Modeling the whole step (lookup, compute window, gradient return, update) on
full batches: does a pool beat the sharded all-to-all status quo, and which of
scheduling / commutative coherence / lookahead gives a real gain over the
CLAY-style uncacheable pool?

## Phases and gates
- **P1 Full-step traffic model** (`step_traffic.py`), full Criteo batches and
  Meta multi-hot batches. Designs: sharded all-to-all (status quo), pool
  uncacheable + owner mailbox, pool MESI + direct update, pool + commutative
  merge. Output: bytes per step per link, bytes on the critical path.
- **P2 Sharing-aware sample placement** (`sample_placement.py`). In
  synchronous training, which GPU runs which sample does not change the
  result. Greedy affinity placement, within a step and across steps.
  Gate: >= 15% fewer row fetches, else drop.
- **P3 Update path in gem5** on a stratified slice (hot + tail rows, real
  sharing degrees): MESI direct read-modify-write (ownership ping-pong),
  owner mailbox (uncacheable and MESI), commutative merge at the pool.
- **P4 Timing with a compute window and lookahead**: rows not written in step
  s are prefetched during its compute; only dependent rows stay on the
  critical path. Compare CLAY, MESI and the best combined design.
- **P5 stretch**: accuracy under bounded staleness of hot rows (real training
  run), or a real commutative state in the Ruby protocol.
- **P6** report with honest novelty notes; commit.

Decision rule: after P1-P2, the proposal is the combination of mechanisms that
each pass their gate. If nothing gains >= 15% step time over the uncacheable
pool, the result is the upper-bound study itself.

## Prior art (novelty risk)
ReCXL (ISCA'24) and TrainingCXL: embedding training inside CXL memory (NMP),
dependency-free prefetch. BagPipe (SOSP'23), ScratchPipe (ISCA'22): lookahead
caches for parameter servers / GPU. Herald (NSDI'24): sample placement by
embedding location over RDMA. COUP (MICRO'15): commutative-update coherence on
multicores. Not found: any of these evaluated for a plain (non-NMP) CXL pool
shared by accelerators with the full training step.
