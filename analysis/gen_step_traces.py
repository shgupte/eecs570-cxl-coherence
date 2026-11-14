#!/usr/bin/env python3
"""Whole-step traces: lookup AND gradient return, on a uniform sample of rows.

Input: per-step caches (keys, mask) from reuse_analysis.py (contiguous sample
split) or sample_placement.py. Rows are sampled uniformly by key hash
(1/--sample), which keeps the real mix of hot and tail rows, sharing degrees
and step-to-step recurrence. Every row is 8 lines.

Each step has 4 phases, each ending in a barrier:
  0 Rp  reads of rows NOT updated in the previous step (with lookahead these
        can be prefetched during the previous step's compute)
  1 Rh  reads of rows updated in the previous step (must wait for it)
  2 W1  every reader publishes its gradient
  3 W2  gradients are combined and the row is updated
Two extra generators (dma0, dma1) only warm the pool before step 0 so that
every later miss is served by the shared L2 (= the pool), in shared state.

Modes (identical read sets; G = gradient of one GPU for one row, 8 lines)
  unc_agg   CLAY-style uncacheable pool. W1: each reader that is not the
            row's combiner stores G to the pool. W2: the combiner (one of the
            row's readers) loads the other readers' G from the pool and
            stores the new row. Transfers: P + 2(P-U) + U.
  comm      Commutative merge at the pool (COUP-style): W1: every reader
            sends its delta once and the pool adds it. Transfers: 2P.
  mesi_rmw  MESI, cached shared rows. W1: every reader applies its update to
            the shared row itself (the line moves from cache to cache).
  mesi_agg  MESI. W1: non-combiners write G to a mailbox line. W2: the
            combiner reads the mailboxes and writes the shared row.

--keep (unc_agg, comm): scheduled software caching. A row whose only reader
in a step is GPU g is updated by g alone, so g's local copy stays current. g
skips the fetch the next time it reads the row, and skips the store while the
row stays private to g (the schedule knows the next reader; the store happens
in the last step before another GPU reads the row, or at once if g does not
read it again within --keep-window steps). This is what MESI gives
for free by leaving the line in M state in g's cache.
Emulation limit: a kept row that needs a store this step is still fetched
first (counted as keep_extra in meta.txt), which overcharges --keep.

Emulation of the uncacheable modes: a GPU reads its k-th row of step s at a
private address (alias s % A, so it never hits its L1) and "stores to the
pool" by dirtying that private copy: an upgrade without data, then one
write-back with data when the line leaves the L1. So a store costs one data
message GPU -> pool, as an uncacheable store would. Mailbox loads in unc_agg
are reads of private never-written lines (one data message pool -> GPU).
"""
import argparse
from pathlib import Path
import numpy as np

LINE, ROW = 64, 512
BASE = 0x40000000
ISSUE_GAP = 1_000
PHASES = 4
MODES = ["unc_agg", "comm", "mesi_rmw", "mesi_agg"]


def lines(bases):
    bases = np.asarray(bases, np.int64).ravel()
    return (bases[:, None] + np.arange(0, ROW, LINE)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--alias", type=int, default=4)
    ap.add_argument("--keep", action="store_true",
                    help="uncacheable modes: software cache of private rows (see docstring)")
    ap.add_argument("--keep-window", type=int, default=4,
                    help="--keep: steps a private row stays in the GPU's cache (match the MESI L1)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    G, A, S = a.gpus, a.alias, a.steps
    unc = a.mode in ("unc_agg", "comm")
    assert unc or not a.keep, "--keep is for the uncacheable modes (MESI keeps lines by itself)"

    steps = []
    for i in range(a.first, a.first + S):
        d = np.load(a.cache / f"b{i}.npz")
        k, m = d["keys"], d["mask"]
        h = (k.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
        sel = (h % np.uint64(a.sample)) == 0
        steps.append((k[sel], m[sel], h[sel].astype(np.int64)))
    allk = np.unique(np.concatenate([k for k, _, _ in steps]))
    NR = len(allk)

    # per step, per GPU: ordered row list (prefetchable first), flags, slots
    plan, prev = [], np.array([], np.int64)
    for s, (k, m, h) in enumerate(steps):
        rid = np.searchsorted(allk, k)
        hz = np.isin(rid, prev)
        bits = (m[:, None] >> np.arange(G)) & 1
        n = bits.sum(1)
        pick = (h + s) % n
        agg = ((np.cumsum(bits, 1) == (pick + 1)[:, None]) & (bits == 1)).argmax(1)
        per = []
        for g in range(G):
            ix = np.flatnonzero(bits[:, g])
            ix = np.concatenate([ix[~hz[ix]], ix[hz[ix]]])
            per.append(dict(ix=ix, rid=rid[ix], hz=hz[ix], isagg=agg[ix] == g))
        plan.append(dict(per=per, rid=rid, bits=bits, n=n, agg=agg, mask=m))
        prev = rid
    # keep: defer[i] = row stays private to its only reader until its next use;
    # ret = this GPU alone updated the row last
    nxt = np.zeros(NR, np.uint8)
    nxt_s = np.full(NR, 1 << 30)
    for s in range(S - 1, -1, -1):
        stp = plan[s]
        near = nxt_s[stp["rid"]] - s <= a.keep_window     # still in the GPU's cache by then
        stp["defer"] = (stp["n"] == 1) & (nxt[stp["rid"]] == stp["mask"]) & near & a.keep
        nxt[stp["rid"]] = stp["mask"]; nxt_s[stp["rid"]] = s
    holder = np.full(NR, -1, np.int8)
    for stp in plan:
        for g, p in enumerate(stp["per"]):
            p["defer"] = stp["defer"][p["ix"]]
            p["skip"] = (holder[p["rid"]] == g) & p["defer"] & a.keep     # no fetch, no store
            p["extra"] = (holder[p["rid"]] == g) & ~p["defer"] & a.keep
        sole = np.where(stp["n"] == 1, stp["bits"].argmax(1), -1).astype(np.int8)
        holder[stp["rid"]] = sole
    maxp = max(len(p["rid"]) for st in plan for p in st["per"])
    maxm = max(int((~p["isagg"]).sum()) for st in plan for p in st["per"])
    maxin = max(int((st["n"][st["agg"] == g] - 1).sum()) for st in plan for g in range(G))
    rnd = lambda x: -(-x * ROW // 4096) * 4096
    STRIDE, MB_STRIDE, IN_STRIDE = rnd(maxp), rnd(maxm), rnd(maxin)

    # contiguous layout: [SHARED][reg g: A x STRIDE][mbin g: A x IN_STRIDE][mb g: MB_STRIDE]
    SHARED = BASE
    reg = [SHARED + NR * ROW + g * A * STRIDE for g in range(G)]
    mbin = [reg[-1] + A * STRIDE + g * A * IN_STRIDE for g in range(G)]
    mb = [mbin[-1] + A * IN_STRIDE + g * MB_STRIDE for g in range(G)]
    end = mb[-1] + MB_STRIDE

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

    # warm-up: both engines read everything the mode will touch
    if unc:
        warm = [np.arange(reg[0], reg[-1] + A * STRIDE, LINE)]
        if a.mode == "unc_agg":
            warm.append(np.arange(mbin[0], mbin[-1] + A * IN_STRIDE, LINE))
    else:
        warm = [np.arange(SHARED, SHARED + NR * ROW, LINE)]
        if a.mode == "mesi_agg":
            warm.append(np.arange(mb[0], end, LINE))
    warm = np.concatenate(warm)
    last = max(issue(G, "r", warm, t), issue(G + 1, "r", warm, t))
    t = barrier(last)
    for _ in range(PHASES - 1):
        t = barrier(t)

    st_ = dict(reads=0, hazard=0, grads=0, mbox=0, upd=0, U=0, extra=0)
    for s, stp in enumerate(plan):
        per = stp["per"]
        copy = lambda g, k: (reg[g] + (s % A) * STRIDE + k * ROW) if unc else SHARED + per[g]["rid"][k] * ROW
        # mailbox slot of each non-combiner pair (mesi_agg), keyed by (row index, gpu)
        slot = {}
        for g in gens:
            for j, i in enumerate(per[g]["ix"][~per[g]["isagg"]]):
                slot[(int(i), g)] = mb[g] + j * ROW
        for ph in (0, 1):                               # Rp, Rh
            last = t
            for g in gens:
                ks = np.flatnonzero((per[g]["hz"] == bool(ph)) & ~per[g]["skip"])
                last = max(last, issue(g, "r", lines(copy(g, ks)), t))
                st_["reads"] += len(ks); st_["hazard"] += len(ks) * ph
                st_["extra"] += int((per[g]["extra"] & (per[g]["hz"] == bool(ph))).sum())
            t = barrier(last)
        last = t                                        # W1: publish gradients
        for g in gens:
            p = per[g]
            if a.mode in ("comm", "mesi_rmw"):
                ks = np.flatnonzero(~p["defer"])
                w = lines(copy(g, ks))
            elif a.mode == "unc_agg":
                ks = np.flatnonzero(~p["isagg"])
                w = lines(copy(g, ks))
            else:
                ks = np.flatnonzero(~p["isagg"])
                w = lines(mb[g] + np.arange(len(ks)) * ROW)
            last = max(last, issue(g, "w", w, t))
            st_["grads"] += len(ks)
        t = barrier(last)
        last = t                                        # W2: combine and update
        if a.mode in ("unc_agg", "mesi_agg"):
            for g in gens:
                p = per[g]
                ks = np.flatnonzero(p["isagg"] & ~p["defer"])
                if a.mode == "unc_agg":
                    nin = int((stp["n"][p["ix"][ks]] - 1).sum())
                    rd = lines(mbin[g] + (s % A) * IN_STRIDE + np.arange(nin) * ROW)
                else:
                    rd = [slot[(int(i), o)] for i in p["ix"][ks] for o in np.flatnonzero(stp["bits"][i]) if o != g]
                    nin = len(rd)
                    rd = lines(rd) if nin else np.array([], np.int64)
                if unc:     # stores first: the mailbox loads then push them out of the L1
                    tt = issue(g, "w", lines(copy(g, ks)), t)
                    last = max(last, issue(g, "r", rd, tt))
                else:
                    tt = issue(g, "r", rd, t)
                    last = max(last, issue(g, "w", lines(copy(g, ks)), tt))
                st_["mbox"] += nin; st_["upd"] += len(ks)
        t = barrier(last)
        st_["U"] += len(stp["rid"])

    tag = a.tag or f"{a.mode}{'_keep' if a.keep else ''}-{a.cache.name}-x{a.sample}"
    d = a.out / tag
    d.mkdir(parents=True, exist_ok=True)
    for g in allg:
        with open(d / (f"gpu{g}.csv" if g < G else f"dma{g - G}.csv"), "w") as f:
            for op, x, tk in sorted(reqs[g], key=lambda q: q[2]):
                f.write(f"{op},{x},{LINE if op != 'b' else 0},{tk}\n")
    meta = (f"mode={a.mode} cache={a.cache.name} sample=1/{a.sample} gpus={G} steps={S} alias={A} "
            f"footprint_MiB={(end - BASE) / 2 ** 20:.1f} rows={NR} barriers={bid} warm_lines={len(warm)} "
            f"per_step: U={st_['U'] / S:.0f} P={st_['reads'] / S:.0f} hazard={st_['hazard'] / S:.0f} "
            f"grads={st_['grads'] / S:.0f} mbox={st_['mbox'] / S:.0f} upd={st_['upd'] / S:.0f} "
            f"keep_extra={st_['extra'] / S:.0f} max_pairs_gpu={maxp}")
    (d / "meta.txt").write_text(meta + "\n")
    print(tag, meta)


if __name__ == "__main__":
    main()
