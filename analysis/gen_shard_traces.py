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
  shard_mig deduplicated, and the schedule moves a row to a GPU that reads
            it when its owner does not: the new owner fetches the row once
            (no gradient to send, it updates locally) and owns it from then
            on. The other readers fetch from the old owner and send their
            gradient to the new one. Ownership evolves from step 0 of the
            cache. This gives sharding the pool's "private rows stay at
            their GPU". Needs --commutative (the gradient goes to another
            address than the fetch).

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
    ap.add_argument("--mode", choices=["shard", "shard_sq", "shard_mig"], required=True)
    ap.add_argument("--batches", type=Path, help="shard_sq: raw batches (b<i>.npz)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--alias", type=int, default=4)
    ap.add_argument("--row-shift", type=int, default=26)
    ap.add_argument("--tag", default="")
    ap.add_argument("--issue-gap", type=int, default=ISSUE_GAP,
                    help="ticks between a GPU's requests (1000 = one line per ns; small = bulk issue)")
    ap.add_argument("--interleave", action="store_true",
                    help="a GPU issues its rows round-robin over the owners (one bulk message per "
                         "peer, all in parallel) instead of in key order (owner after owner)")
    a = ap.parse_args()
    globals()["ISSUE_GAP"] = a.issue_gap
    G, A, S = a.gpus, a.alias, a.steps
    files = sorted(a.cache.glob("b*.npz"))
    tlen = np.max([np.load(f)["rowmax"] for f in files], axis=0) + 1
    hsh = lambda k: (k.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)

    plan, prev = [], np.array([], np.int64)
    st_ = dict(U=0, P=0, local=0, xfer=0, hazard=0, grads=0, moved=0)
    mig = a.mode == "shard_mig"
    cur = {}                                            # shard_mig: current owner of moved rows
    for i in range(0 if mig else a.first, a.first + S):
        d = np.load(a.cache / f"b{i}.npz")
        k, m = d["keys"], d["mask"]
        sel = (hsh(k) % np.uint64(a.sample)) == 0
        k, m = k[sel], m[sel]
        t, r = k >> a.row_shift, k & ((1 << a.row_shift) - 1)
        own = (r // -(-tlen[t] // G)).astype(np.int64)
        hz = np.isin(k, prev)
        bits = (m[:, None] >> np.arange(G)) & 1
        if mig:
            own = np.array([cur.get(int(x), int(o)) for x, o in zip(k, own)], np.int64)
            stay = bits[np.arange(len(k)), own] == 1
            pick = (hsh(k).astype(np.int64) + i) % bits.sum(1)
            first = ((np.cumsum(bits, 1) == (pick + 1)[:, None]) & (bits == 1)).argmax(1)
            new = np.where(stay, own, first)
            for x, o in zip(k[~stay], new[~stay]):
                cur[int(x)] = int(o)
            if i < a.first:
                prev = k
                continue
            per = []
            for g in range(G):
                rd = np.flatnonzero((bits[:, g] == 1) & ((new != g) | ~stay))   # fetch from the old owner
                rd = np.concatenate([rd[~hz[rd]], rd[hz[rd]]])
                wr = np.flatnonzero((bits[:, g] == 1) & (new != g))             # gradient to the new owner
                o, wo = own[rd], new[wr]
                j = np.zeros(len(rd), np.int64); wj = np.zeros(len(wr), np.int64)
                for x in range(G):
                    j[o == x] = np.arange(int((o == x).sum()))
                    wj[wo == x] = np.arange(int((wo == x).sum()))
                per.append(dict(o=o, j=j, hz=hz[rd], wo=wo, wj=wj))
                st_["xfer"] += len(rd); st_["hazard"] += int(hz[rd].sum()); st_["grads"] += len(wr)
            st_["U"] += len(k); st_["P"] += int(bits.sum()); st_["moved"] += int((~stay).sum())
            st_["local"] += int(stay.sum())
            plan.append(per)
            prev = k
            continue
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
    maxj = max(int(p[x].max()) + 1 for per in plan for p in per for x in ("j", "wj") if x in p and len(p[x]))
    maxrows = max(len(p["j"]) for per in plan for p in per)
    STRIDE = maxj * BLK
    reg = [BASE + g * A * STRIDE for g in range(G)]
    end = reg[-1] + A * STRIDE
    wreg = [end + g * A * STRIDE for g in range(G)]     # shard_mig: gradient addresses
    if mig:
        end = wreg[-1] + A * STRIDE

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
        def rr(sel, j):                                 # round-robin over owners
            ix = np.flatnonzero(sel)
            return ix[np.argsort(j[ix], kind="stable")] if a.interleave else ix
        for ph in (0, 1):
            last = t
            for g in gens:
                last = max(last, issue(g, "r", lines(g, rr(per[g]["hz"] == bool(ph), per[g]["j"])), t))
            t = barrier(last)
        last = t
        for g in gens:
            if mig:
                p = per[g]
                o_ = rr(np.ones(len(p["wj"]), bool), p["wj"])
                base = wreg[g] + (s % A) * STRIDE + p["wj"][o_] * BLK + p["wo"][o_] * LINE
                w = (base[:, None] + np.arange(0, BLK, ROW)).ravel()
            else:
                w = lines(g, rr(np.ones(len(per[g]["j"]), bool), per[g]["j"]))
            last = max(last, issue(g, "w", w, t))
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
            f"grads={(st_['grads'] if mig else st_['xfer']) / S:.0f} moved={st_['moved'] / S:.0f} "
            f"max_rows_gpu={maxrows} l1_min_KiB={maxrows * ROW // 1024}")
    (d / "meta.txt").write_text(meta + "\n")
    print(tag, meta)


if __name__ == "__main__":
    main()
