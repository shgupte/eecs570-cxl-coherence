#!/usr/bin/env python3
"""Whole-step traces for the status quo: tables sharded over the GPUs'
own memory, all-to-all exchange (ZionEX / TorchRec style). Same row sample,
steps, phases and barriers as gen_step_traces.py, so summarize_step.py reads
both. Replay with `configs/cxl-trace.py --topology sharded`: there are G L2
controllers, L2 i is GPU i's shard and sits behind GPU i's port.

A row lives on a static owner GPU (row-wise blocks, as in step_traffic.py).
Rows a GPU owns are free (local memory, not modeled). For every other row:
  Rp/Rh  the reader fetches the row from the owner (prefetchable / must wait
         for the previous step's update, as in the pool traces)
  W1     the reader returns its gradient to the owner, which updates the row
         locally (W2 is empty)
Modes
  shard     one fetch and one gradient per (GPU,row) pair (deduplicated)
  shard_sq  one fetch and one gradient per lookup (what TorchRec sends for
            one-hot features); needs --batches for the per-GPU lookup counts

Emulation: as the uncacheable pool modes. The reader uses a private address
whose bits 6..8 name the owner (the protocol picks the L2 by these bits), so
a row is 8 lines 512 B apart. The gradient is a store to the fetched copy
and one write-back when the line leaves the L1. The L1 must hold one step of
a GPU's rows (meta.txt: l1_min_KiB), or stores would refetch.
"""
import argparse
from pathlib import Path
import numpy as np

LINE, ROW, BLK = 64, 512, 4096
BASE = 0x40000000
ISSUE_GAP = 1_000
PHASES = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["shard", "shard_sq"], required=True)
    ap.add_argument("--batches", type=Path, help="shard_sq: raw batches (b<i>.npz)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--alias", type=int, default=4)
    ap.add_argument("--row-shift", type=int, default=26)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    G, A, S = a.gpus, a.alias, a.steps
    files = sorted(a.cache.glob("b*.npz"))
    tlen = np.max([np.load(f)["rowmax"] for f in files], axis=0) + 1
    hsh = lambda k: (k.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)

    plan, prev = [], np.array([], np.int64)
    st_ = dict(U=0, P=0, local=0, xfer=0, hazard=0)
    for i in range(a.first, a.first + S):
        d = np.load(a.cache / f"b{i}.npz")
        k, m = d["keys"], d["mask"]
        sel = (hsh(k) % np.uint64(a.sample)) == 0
        k, m = k[sel], m[sel]
        t, r = k >> a.row_shift, k & ((1 << a.row_shift) - 1)
        own = (r // -(-tlen[t] // G)).astype(np.int64)
        hz = np.isin(k, prev)
        bits = (m[:, None] >> np.arange(G)) & 1
        cnt = bits.astype(np.int64)                     # transfers per (row, GPU)
        if a.mode == "shard_sq":
            b = np.load(a.batches / f"b{i}.npz")
            lengths = b["lengths"]
            T, B = lengths.shape
            valid = lengths > 0
            gpu = d["gpu"].astype(np.int64) if "gpu" in d else np.arange(B) // (B // G)
            lk = (np.repeat(np.arange(T, dtype=np.int64), valid.sum(1)) << a.row_shift) | b["indices"]
            lg = np.broadcast_to(gpu, (T, B))[valid]
            ix = np.searchsorted(k, lk); ix[ix >= len(k)] = 0
            hit = k[ix] == lk
            cnt = np.zeros((len(k), G), np.int64)
            np.add.at(cnt, (ix[hit], lg[hit]), 1)
            assert ((cnt > 0) == (bits > 0)).all()
        per = []
        for g in range(G):
            rows = np.flatnonzero((bits[:, g] == 1) & (own != g))
            rows = np.repeat(rows, cnt[rows, g])
            rows = np.concatenate([rows[~hz[rows]], rows[hz[rows]]])
            o = own[rows]
            j = np.zeros(len(rows), np.int64)           # index within (g, owner)
            for x in range(G):
                j[o == x] = np.arange(int((o == x).sum()))
            per.append(dict(o=o, j=j, hz=hz[rows]))
            st_["xfer"] += len(rows); st_["hazard"] += int(hz[rows].sum())
        st_["U"] += len(k); st_["P"] += int(bits.sum())
        st_["local"] += int(bits[np.arange(len(k)), own].sum())
        plan.append(per)
        prev = k
    maxj = max(int(p["j"].max()) + 1 for per in plan for p in per if len(p["j"]))
    maxrows = max(len(p["j"]) for per in plan for p in per)
    STRIDE = maxj * BLK
    reg = [BASE + g * A * STRIDE for g in range(G)]
    end = reg[-1] + A * STRIDE

    gens = list(range(G))
    allg = gens + [G, G + 1]
    reqs = {g: [] for g in allg}
    t, bid = 0, 0

    def issue(g, op, addrs, t0):
        addrs = np.asarray(addrs, np.int64)
        reqs[g].extend((op, int(x), t0 + 1 + i * ISSUE_GAP) for i, x in enumerate(addrs))
        return t0 + 1 + max(len(addrs) - 1, 0) * ISSUE_GAP if len(addrs) else t0

    def barrier(last):
        nonlocal bid
        for g in allg:
            reqs[g].append(("b", bid, last + 1))
        bid += 1
        return last + 2

    warm = np.arange(BASE, end, LINE)
    last = max(issue(G, "r", warm, t), issue(G + 1, "r", warm, t))
    t = barrier(last)
    for _ in range(PHASES - 1):
        t = barrier(t)

    for s, per in enumerate(plan):
        def lines(g, sel):
            p = per[g]
            base = reg[g] + (s % A) * STRIDE + p["j"][sel] * BLK + p["o"][sel] * LINE
            return (base[:, None] + np.arange(0, BLK, ROW)).ravel()
        for ph in (0, 1):
            last = t
            for g in gens:
                last = max(last, issue(g, "r", lines(g, per[g]["hz"] == bool(ph)), t))
            t = barrier(last)
        last = t
        for g in gens:
            last = max(last, issue(g, "w", lines(g, np.ones(len(per[g]["j"]), bool)), t))
        t = barrier(last)
        t = barrier(t)

    tag = a.tag or f"{a.mode}-{a.cache.name}-x{a.sample}"
    d = a.out / tag
    d.mkdir(parents=True, exist_ok=True)
    for g in allg:
        with open(d / (f"gpu{g}.csv" if g < G else f"dma{g - G}.csv"), "w") as f:
            for op, x, tk in sorted(reqs[g], key=lambda q: q[2]):
                f.write(f"{op},{x},{LINE if op != 'b' else 0},{tk}\n")
    meta = (f"mode={a.mode} cache={a.cache.name} sample=1/{a.sample} gpus={G} steps={S} alias={A} "
            f"footprint_MiB={(end - BASE) / 2 ** 20:.1f} barriers={bid} warm_lines={len(warm)} "
            f"per_step: U={st_['U'] / S:.0f} P={st_['P'] / S:.0f} local={st_['local'] / S:.0f} "
            f"fetches={st_['xfer'] / S:.0f} hazard={st_['hazard'] / S:.0f} "
            f"max_rows_gpu={maxrows} l1_min_KiB={maxrows * ROW // 1024}")
    (d / "meta.txt").write_text(meta + "\n")
    print(tag, meta)


if __name__ == "__main__":
    main()
