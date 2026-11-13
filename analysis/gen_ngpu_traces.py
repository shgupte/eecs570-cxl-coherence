#!/usr/bin/env python3
"""N-GPU embedding traces for comparing MESI with no-coherence designs.

Every generator g = GPU g (default 8). GPU g reads the hot rows it used in each
step (measured Criteo masks) and owns rows with owner == g. At each write
boundary (every --stale k steps) every owner writes each row it owns that any
GPU touched in the window; --write-frac p < 1 writes only a deterministic
fraction p of them (read-mostly rows: frozen / lazily updated tables).

Each step has 4 phases, each ending in a barrier (4 per step, always):
  0 read  : forward pass, the critical path
  1 write : owners update their rows
  2 sync1 : mode-specific publish (write-backs, lists, pushes)
  3 sync2 : mode-specific consume (swinv list reads)
Barrier 4s ends the read phase of step s, 4s+3 ends the step. Alias modes get
one warm-up step first (barriers 0..3), so all misses later hit the shared L2
as MESI re-reads do.

Modes (identical read requests in every mode):
  mesi     one shared copy per row; hardware MESI keeps the L1s coherent.
  uncached CLAY-style uncacheable pool: every read goes to the pool every
           step (fresh alias each step, so the L1 never hits).
  swflush  software coherence, flush-all: GPUs cache rows; at each boundary
           every GPU drops everything it cached (all copies change address)
           and owners write their updated rows back to the pool (sync1: one
           write per updated line).
  swinv    software coherence, selective: owners write back (as swflush) and
           publish the ids of updated rows (8 B each, sync1); after a barrier
           every GPU reads every list (sync2) and drops only those rows.
           Unchanged rows stay cached: hardware's precise invalidation done
           in software.
  update   ideal write-update protocol (= owner push of new rows, Meta-style):
           readers' copies are kept fresh so held rows always hit; sync1:
           for each updated row the owner sends every line to every other
           GPU that used the row in the window (one write per line per
           holder). Best case for any coherence protocol.

Software-mode emulation: GPU g addresses copy (row, gen % A) in its own
region, gen = epoch (swflush) or row version (swinv). A new gen misses in L1
and hits in L2. False L1 hits need A gens between two reads of a row by one
GPU; such reads are counted (risky_reads in meta.txt).
"""
import argparse
from pathlib import Path
import numpy as np

LINE, ROW = 64, 512
LPR = ROW // LINE
BASE = 0x40000000
ISSUE_GAP = 1_000           # 1 ns between issues; --max-outstanding throttles
PHASES = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--first-batch", type=int, default=0)
    ap.add_argument("--stale", type=int, default=1)
    ap.add_argument("--write-frac", type=float, default=1.0)
    ap.add_argument("--mode", choices=["mesi", "uncached", "swflush", "swinv", "update"], required=True)
    ap.add_argument("--alias", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    G, A = a.gpus, a.alias

    h = np.load(a.hot)
    sm = h["steps_mask"][a.first_batch:a.first_batch + a.steps, : a.rows].astype(np.uint16)
    owner = h["owner"][: a.rows] % G
    S, R = sm.shape
    rows = np.arange(R)
    gens = list(range(G))
    # Write-backs (sw modes) and pushes (update) are issued by a per-GPU copy
    # engine (generator G+g, own L1) so they do not evict the GPU's cached rows.
    dma = a.mode in ("swflush", "swinv", "update")
    allg = gens + ([G + g for g in gens] if dma else [])
    # Contiguous layout (no power-of-two strides, so no L2 set conflicts):
    # [SHARED][GPU0 A copies]..[GPU G-1][WB A copies][LIST 2 x G x 64 KiB][PUSH A x G copies]
    ALIAS_STRIDE = R * ROW
    REGION = ALIAS_STRIDE if a.mode in ("mesi", "update") else A * ALIAS_STRIDE
    SHARED = BASE
    reg = [BASE + ALIAS_STRIDE + g * REGION for g in gens]
    WB = BASE + ALIAS_STRIDE + G * REGION
    LIST = WB + REGION
    PUSH = LIST + 2 * G * 65536
    DUMMY = PUSH + (A * G * ALIAS_STRIDE if a.mode == "update" else 0)
    end = DUMMY + (2 << 20)
    assert end - BASE <= (1 << 30), "footprint over 1 GiB"
    sw = a.mode in ("swflush", "swinv")
    version = np.zeros(R, np.int64)
    key = np.arange(R, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)

    reqs = {g: [] for g in allg}
    t, bid = 0, 0

    def lines(bases):
        bases = np.asarray(bases, np.int64).ravel()
        return (bases[:, None] + np.arange(0, ROW, LINE)).ravel()

    def issue(g, op, addrs, t0):
        addrs = np.asarray(addrs, np.int64)
        reqs[g].extend((op, int(x), t0 + 1 + i * ISSUE_GAP) for i, x in enumerate(addrs))
        return t0 + 1 + max(len(addrs) - 1, 0) * ISSUE_GAP

    def barrier(last):
        nonlocal bid
        for g in allg:
            reqs[g].append(("b", bid, last + 1))
        bid += 1
        return last + 2

    def gen_of(rs, epoch):
        return np.full(len(rs), epoch) if a.mode == "swflush" else version[rs]

    def copy_addr(g, rs, s, epoch):
        if a.mode == "mesi":
            return SHARED + rs * ROW
        if a.mode == "uncached":
            return reg[g] + (s % A) * ALIAS_STRIDE + rs * ROW
        if a.mode == "update":
            return reg[g] + rs * ROW
        return reg[g] + (gen_of(rs, epoch) % A) * ALIAS_STRIDE + rs * ROW

    def push_addr(d, rs, epoch):
        return PUSH + ((epoch % A) * G + d) * (R * ROW) + rs * ROW

    # warm-up: each GPU first touches the write-back / push lines it will write
    # (so later RFOs hit L2, not DRAM), then loads GPU (g+1)'s read aliases,
    # which also evicts its own warm-up lines from its L1.
    # Private copies of the alias modes are read by TWO other GPUs, one after
    # the other, so the L2 holds them in shared state: a reader then gets S,
    # not E, and its clean evictions are silent. (In MESI_Two_Level an E line
    # is written back with data on eviction even when clean; a software-managed
    # cache would drop it silently, as MESI does with its S lines.)
    if a.mode in ("uncached", "swflush", "swinv", "update"):
        nwarm = 0
        rounds = 1 if a.mode == "update" else A
        for e in range(rounds):
            for part in (1, 2):
                if part == 2 and a.mode == "update":
                    continue
                last = t
                for g in gens:
                    if e == 0 and part == 1 and dma:
                        own = rows[owner == g]
                        eng = []
                        if sw:
                            eng.append(lines(np.concatenate([WB + x * ALIAS_STRIDE + own * ROW for x in range(A)])))
                        if a.mode == "update":
                            eng.append(lines(np.concatenate([push_addr(d, own, x) for x in range(A) for d in gens])))
                        eng.append(DUMMY + np.arange(0, 2 << 20, LINE))   # evict them from the engine's L1
                        last = max(last, issue(G + g, "r", np.concatenate(eng), t))
                    base = reg[g] + (0 if a.mode == "update" else e * ALIAS_STRIDE)
                    last = max(last, issue((g + part) % G, "r", lines(base + rows * ROW), t))
                t = barrier(last)
                nwarm += 1
        while nwarm % PHASES:
            t = barrier(t)
            nwarm += 1

    st = dict(reads=0, writes=0, sync=0, risky=0)
    last_gen = np.full((G, R), -1, np.int64)
    window = np.zeros(R, np.uint16)
    for s in range(S):
        epoch, boundary = s // a.stale, (s + 1) % a.stale == 0
        last = t                                        # read phase
        for g in gens:
            rs = np.flatnonzero(sm[s] & (1 << g))
            if sw:
                cur, lg = gen_of(rs, epoch), last_gen[g, rs]
                st["risky"] += int(((lg >= 0) & (cur != lg) & ((cur - lg) % A == 0)).sum())
                last_gen[g, rs] = cur
            last = max(last, issue(g, "r", lines(copy_addr(g, rs, s, epoch)), t))
            st["reads"] += len(rs) * LPR
        window |= sm[s]
        t = barrier(last)

        written = np.array([], np.int64)               # write phase
        last = t
        if boundary:
            written = np.flatnonzero(window)
            if a.write_frac < 1:
                hs = (key[written] ^ np.uint64((epoch + 1) * 0x85EBCA6B)) * np.uint64(0xC2B2AE3D27D4EB4F)
                written = written[(hs >> np.uint64(40)).astype(np.float64) / 2 ** 24 < a.write_frac]
            for g in gens:
                ws = written[owner[written] == g]
                last = max(last, issue(g, "w", lines(copy_addr(g, ws, s, epoch)), t))
                st["writes"] += len(ws) * LPR
        t = barrier(last)

        last = t                                        # sync1: publish
        nlist = [-(-int((owner[written] == o).sum()) * 8 // LINE) for o in gens]
        if boundary and len(written):
            for g in gens:
                ws = written[owner[written] == g]
                out = []
                if sw:
                    out.append(lines(WB + (epoch % A) * ALIAS_STRIDE + ws * ROW))
                if a.mode == "update":
                    for d in gens:
                        if d != g:
                            tgt = ws[(window[ws] & (1 << d)) != 0]
                            out.append(lines(push_addr(d, tgt, epoch)))
                out = np.concatenate(out) if out else np.array([], np.int64)
                if dma:
                    last = max(last, issue(G + g, "w", out, t))   # copy engine
                st["sync"] += len(out)
                if a.mode == "swinv":                        # GPU writes its small id list
                    lst = LIST + ((epoch % 2) * G + g) * 65536 + np.arange(nlist[g]) * LINE
                    last = max(last, issue(g, "w", lst, t))
                    st["sync"] += len(lst)
        t = barrier(last)

        last = t                                        # sync2: consume
        if boundary and len(written) and a.mode == "swinv":
            allists = np.concatenate([LIST + ((epoch % 2) * G + o) * 65536 + np.arange(nlist[o]) * LINE
                                      for o in gens])
            for g in gens:
                last = max(last, issue(g, "r", allists, t))
                st["sync"] += len(allists)
        t = barrier(last)
        if boundary:
            version[written] += 1
            window[:] = 0

    tag = a.tag or (f"{a.mode}-g{G}-r{R}-k{a.stale}" + (f"-p{a.write_frac:g}" if a.write_frac < 1 else "")
                    + (f"-b{a.first_batch}" if a.first_batch else ""))
    d = a.out / tag
    d.mkdir(parents=True, exist_ok=True)
    for g in allg:
        with open(d / (f"gpu{g}.csv" if g < G else f"dma{g - G}.csv"), "w") as f:
            for op, x, tk in sorted(reqs[g], key=lambda q: q[2]):
                f.write(f"{op},{x},{LINE if op != 'b' else 0},{tk}\n")
    footprint_mb = (end - BASE) / 2 ** 20
    meta = (f"footprint_MiB={footprint_mb:.1f} mode={a.mode} gpus={G} rows={R} steps={S} stale={a.stale} write_frac={a.write_frac} "
            f"alias={A} barriers={bid} reads={st['reads']} writes={st['writes']} sync_lines={st['sync']} "
            f"risky_reads={st['risky']}")
    (d / "meta.txt").write_text(meta + "\n")
    print(tag, meta)


if __name__ == "__main__":
    main()
