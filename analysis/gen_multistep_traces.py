#!/usr/bin/env python3
"""Multi-step hot-row replay traces from measured sharing (reuse_analysis.py).

Three trace generators:
  accel0 = GPU 0, accel1 = GPU 1 (readers, private L1 each),
  owner  = the updater for every row whose owner is not GPU 0 or 1.
Rows owned by GPU 0/1 are updated by that accelerator itself.

Per step s (measured masks for batch s):
  read phase : accelN reads all 8 lines of each hot row GPU N touched in s
  barrier
  write phase: each row touched in s by GPU 0 or 1 is written once by its updater
  barrier
Barriers ("b,<id>,0,<tick>") need the patched CsvTraceGen: each generator
waits for its own outstanding requests and for every other generator, so the
barrier release ticks give real phase and step times.

--mode shared  : one copy of every row; coherence acts on it.
--mode sharded : every generator works in its own address region (reads and
                 writes), so no line is ever touched by two caches. Same
                 request counts and order. shared minus sharded is the
                 whole coherence cost.
--mode push    : selective update-push emulation. After the write phase,
                 every reader re-reads the rows it holds (rows it has read in
                 an earlier step); the invalidate+refetch stands in for the
                 owner pushing the new value. The next read phase then hits.
--mode uncached: no-coherence baseline (CLAY-style uncacheable pool). Every
                 access by generator g in step s goes to a fresh alias
                 (g, s) of the row, so no line is ever reused from a cache
                 and no coherence message is ever sent: each read and write
                 crosses CXL to pooled memory every step.
--stale S      : the owner writes (and pushes) only every S steps; readers
                 keep reading their bounded-stale copies in between.
Every step has three phases (read, write, push) and three barriers, so
barrier 3s ends the read phase of step s (the critical path), 3s+1 the write
phase and 3s+2 the push phase, in every mode.
--no-barriers  : legacy open-loop timing (v1 behaviour) for comparison.
"""
import argparse
from pathlib import Path
import numpy as np

LINE, ROW = 64, 512
BASE = 0x40000000
REGION = 256 * 1024 * 1024
ISSUE_GAP = 1_000          # ticks between issues (1 ns); 32-outstanding cap throttles
DRAIN_PER_LINE = 50_000    # only used with --no-barriers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--mode", choices=["shared", "sharded", "push", "uncached"], default="shared")
    ap.add_argument("--no-barriers", action="store_true")
    ap.add_argument("--stale", type=int, default=1)
    ap.add_argument("--alias", type=int, default=1000, help="uncached: aliases cycle every N steps (4 lets L2 hit, L1 miss)")
    a = ap.parse_args()
    h = np.load(a.hot)
    sm = h["steps_mask"][: a.steps]
    keep = np.flatnonzero((sm & 0b11).any(0))[: a.rows]
    sm, owner = sm[:, keep], h["owner"][keep]
    addr = BASE + np.arange(len(keep), dtype=np.int64) * ROW
    gens = (0, 1, 2)
    region = {g: (g * REGION if a.mode == "sharded" else 0) for g in gens}
    reqs = {g: [] for g in gens}
    t = 0
    bid = 0
    stats = []

    STEP_ALIAS = 8 * 1024 * 1024
    cur = {"s": 0}
    if a.mode == "uncached":
        region = {g: g * REGION for g in gens}

    def lines_of(rows, g):
        off = region[g] + ((cur["s"] % a.alias) * STEP_ALIAS if a.mode == "uncached" else 0)
        return (addr[rows, None] + off + np.arange(0, ROW, LINE)).ravel()

    def barrier(t, n_lines, drain):
        nonlocal bid
        if a.no_barriers:
            return t + n_lines * drain + 1
        for g in gens:
            reqs[g].append(("b", bid, t))
        bid += 1
        return t + 1

    held = {0: np.zeros(len(keep), bool), 1: np.zeros(len(keep), bool)}
    for s in range(sm.shape[0]):
        cur["s"] = s
        r0 = np.flatnonzero(sm[s] & 1)
        r1 = np.flatnonzero(sm[s] & 2)
        last = t
        for g, rows in ((0, r0), (1, r1)):
            ls = lines_of(rows, g)
            reqs[g] += [("r", int(x), t + 1 + i * ISSUE_GAP) for i, x in enumerate(ls)]
            if len(ls):
                last = max(last, t + 1 + (len(ls) - 1) * ISSUE_GAP)
        n = max(len(r0), len(r1)) * (ROW // LINE)
        t = barrier(last + 1, n, DRAIN_PER_LINE)
        held[0][r0] = True
        held[1][r1] = True
        update = (s + 1) % a.stale == 0
        touched = np.union1d(r0, r1) if update else np.array([], dtype=np.int64)
        wn = {}
        last = t
        for g in gens:
            rows = touched[(owner[touched] == g) if g < 2 else (owner[touched] >= 2)]
            ls = lines_of(rows, g)
            reqs[g] += [("w", int(x), t + 1 + i * ISSUE_GAP) for i, x in enumerate(ls)]
            if len(ls):
                last = max(last, t + 1 + (len(ls) - 1) * ISSUE_GAP)
            wn[g] = len(rows)
        t = barrier(last + 1, len(touched) * (ROW // LINE), DRAIN_PER_LINE // 4)
        last = t
        npush = 0
        if a.mode == "push" and update:
            for g in (0, 1):
                rows = np.flatnonzero(held[g])
                ls = lines_of(rows, g)
                reqs[g] += [("r", int(x), t + 1 + i * ISSUE_GAP) for i, x in enumerate(ls)]
                if len(ls):
                    last = max(last, t + 1 + (len(ls) - 1) * ISSUE_GAP)
                npush += len(rows)
        t = barrier(last + 1, npush * (ROW // LINE), DRAIN_PER_LINE)
        stats.append((s, len(r0), len(r1), len(np.intersect1d(r0, r1)), wn[0], wn[1], wn[2]))
    tag = a.mode + ("" if a.stale == 1 else f"-s{a.stale}") + ("" if a.alias == 1000 else f"-a{a.alias}")
    a.out.mkdir(parents=True, exist_ok=True)
    for g, name in ((0, "accel0"), (1, "accel1"), (2, "owner")):
        with open(a.out / f"{tag}-{name}.csv", "w") as f:
            for op, x, tk in sorted(reqs[g], key=lambda q: q[2]):
                f.write(f"{op},{x},{LINE if op != 'b' else 0},{tk}\n")
    with open(a.out / f"{tag}-steps.csv", "w") as f:
        f.write("step,rows_gpu0,rows_gpu1,rows_both,writes_by_gpu0,writes_by_gpu1,writes_by_other_owner\n")
        for r in stats:
            f.write(",".join(map(str, r)) + "\n")
    print(f"rows={len(keep)} steps={sm.shape[0]} barriers={bid} reqs="
          f"{[len(v) for v in reqs.values()]} last_file_tick={t}")


if __name__ == "__main__":
    main()
