#!/usr/bin/env python3
"""Link-level model of one DLRM training step (lookup, gradient return, row
update) for the sharded all-to-all designs and the CXL pool designs.

Why: gem5 replays a 1/64 row sample with one request per 64 B line. With bulk
transfers every design is limited by link bandwidth, so the step time is set
by how many bytes cross which link. This model computes exactly those bytes
from the same per-step caches (keys, GPU mask) the trace generators read, at
any sample rate including the full batch, and shares each link's bandwidth
among the transfers that use it (max-min fair, fluid). It is validated
against the gem5 bulk runs with --bytes gem5 (same message sizes).

Fabric
  every GPU has an up and a down link of --gpu-bw GB/s (sharded designs:
  --nv-bw, the GPU-to-GPU fabric); the pool has an in and an out link of
  --pool-bw GB/s. GPU a -> GPU b uses up[a], down[b]; GPU -> pool uses up[g],
  pool-in; pool -> GPU uses pool-out, down[g].
Phases (as the traces): Rp, Rh, W1, W2, each ends when its last transfer
  ends, plus --lat-us per non-empty phase.
Bytes per row
  gem5  request 8 x 8 B, data 8 x 72 B, ack 8 x 8 B (Ruby message sizes)
  phys  sharded: 8 B index, row payload, no per-row ack (bulk messages);
        pool: per 64 B line --cxl-req / --cxl-data / --cxl-ack bytes
Designs
  shard_sq, shard, shard_mig      gen_shard_traces.py (rows travel)
  pooled, pooled_dedup            table-wise sharding, pooled vectors travel (see pooled_flows)
  nmp, nmp_dedup                  CXL pool with near-memory processing: the pool sums a bag
                                  and returns one vector; one gradient vector per bag goes
                                  back and the pool applies it to the bag's rows. Stats
                                  uniq_rows / lookups = rows the pool touches per step.
  unc_agg, comm (+ _keep)         gen_step_traces.py
  with --pool-bulk, comm / comm_keep are the host-DRAM hybrid (rows in server
  memory, GPUs fetch and return gradients over PCIe; _keep = GPU-side cache
  driven by lookahead, as in BagPipe / ScratchPipe)
  mesi_pm                         plain MESI, every transfer through the pool (CXL has
                                  no cache-to-cache forwarding)
  mesi_comm                       MESI through the pool + commutative updates
                                  (Downgrade leaves no copy at the old owner;
                                  --stale reproduces the first protocol
                                  version, where it kept a shared copy that
                                  later deltas did not invalidate)
"""
import argparse
from pathlib import Path
import numpy as np

C64 = np.uint64(0x9E3779B97F4A7C15)
hsh = lambda k: (k.astype(np.uint64) * C64) >> np.uint64(40)


def load(cache, first, n, sample):
    out = []
    for i in range(first, first + n):
        f = cache / f"b{i}.npz"
        if not f.exists():
            break
        d = np.load(f)
        k, m, h = d["keys"], d["mask"], hsh(d["keys"])
        if sample > 1:
            sel = (h % np.uint64(sample)) == 0
            k, m, h = k[sel], m[sel], h[sel]
        out.append(dict(i=i, k=k, m=m, h=h.astype(np.int64), gpu=d["gpu"] if "gpu" in d else None))
    return out


def fluid(nbytes, A, cap):
    """Time for all flows to finish; flows share links max-min fairly."""
    rem = nbytes.astype(float).copy()
    act = rem > 0
    t = 0.0
    while act.any():
        rate = np.zeros(len(rem))
        capl = cap.astype(float).copy()
        un = act.copy()
        while un.any():
            cnt = A[un].sum(0)
            share = np.where(cnt > 0, capl / np.maximum(cnt, 1), np.inf)
            l = int(share.argmin())
            sel = un & A[:, l]
            rate[sel] = share[l]
            capl -= A[sel].sum(0) * share[l]
            un &= ~sel
        dt = (rem[act] / rate[act]).min()
        rem[act] -= rate[act] * dt
        t += dt
        act = rem > 1e-9
    return t


class Fabric:
    def __init__(self, G, gpu_bw, pool_bw, lat_us):
        self.G, self.lat = G, lat_us
        # links: up[g]=g, down[g]=G+g, pool-in=2G, pool-out=2G+1; GB/s = 1000 B/us
        self.cap = np.array([gpu_bw] * (2 * G) + [pool_bw, pool_bw], float) * 1000.0
        pairs = [(s, d) for s in range(G + 1) for d in range(G + 1) if s != d]
        self.pairs = pairs
        self.A = np.zeros((len(pairs), 2 * G + 2), bool)
        for j, (s, d) in enumerate(pairs):
            self.A[j, s if s < G else 2 * G + 1] = True
            self.A[j, G + d if d < G else 2 * G] = True

    def time(self, F):
        nb = np.array([F[s, d] for s, d in self.pairs])
        return (fluid(nb, self.A, self.cap) + self.lat) if nb.sum() > 0 else 0.0


def bits_of(m, G):
    return ((m[:, None] >> np.arange(G)) & 1).astype(bool)


def add_pair(F, src, dst, w):
    """F[src[i], dst[i]] += w[i] (arrays)."""
    np.add.at(F, (np.asarray(src), np.asarray(dst)), w)


def shard_flows(a, B, mode):
    G = a.gpus
    files = sorted(a.cache.glob("b*.npz"))
    tlen = np.max([np.load(f)["rowmax"] for f in files], axis=0) + 1
    mig = mode == "shard_mig"
    steps = load(a.cache, 0 if mig else a.first, a.first + a.steps if mig else a.steps, a.sample)
    allk = np.unique(np.concatenate([s["k"] for s in steps]))
    cur = np.full(len(allk), -1, np.int64)
    out, prev, stats = [], np.array([], np.int64), dict(moved=0, fetch=0, grads=0)
    for s in steps:
        k, i = s["k"], s["i"]
        bits = bits_of(s["m"], G)
        t, r = k >> a.row_shift, k & ((1 << a.row_shift) - 1)
        own = (r // -(-tlen[t] // G)).astype(np.int64)
        hz = np.isin(k, prev)
        prev = k
        F = np.zeros((4, G + 1, G + 1))
        U = np.arange(len(k))
        if mig:
            rid = np.searchsorted(allk, k)
            own = np.where(cur[rid] >= 0, cur[rid], own)
            stay = bits[U, own]
            pick = (s["h"] + i) % bits.sum(1)
            first = ((np.cumsum(bits, 1) == (pick + 1)[:, None]) & bits).argmax(1)
            new = np.where(stay, own, first)
            cur[rid[~stay]] = new[~stay]
            if i < a.first:
                continue
            for g in range(G):
                rd = bits[:, g] & ((new != g) | ~stay)
                wr = bits[:, g] & (new != g)
                for ph in (0, 1):
                    x = rd & (hz == bool(ph))
                    c = np.bincount(own[x], minlength=G).astype(float)
                    F[ph, g, :G] += c * B["req"]; F[ph, :G, g] += c * B["data"]
                c = np.bincount(new[wr], minlength=G).astype(float)
                F[2, g, :G] += c * B["data"]; F[2, :G, g] += c * B["ack"]
                stats["fetch"] += int(rd.sum()); stats["grads"] += int(wr.sum())
            stats["moved"] += int((~stay).sum())
        else:
            cnt = bits.astype(np.int64)
            if mode == "shard_sq":
                b = np.load(a.batches / f"b{i}.npz")
                lengths = b["lengths"]
                T, Bn = lengths.shape
                valid = lengths > 0
                gpu = s["gpu"].astype(np.int64) if s["gpu"] is not None else np.arange(Bn) // (Bn // G)
                lk = (np.repeat(np.arange(T, dtype=np.int64), valid.sum(1)) << a.row_shift) | b["indices"]
                lg = np.broadcast_to(gpu, (T, Bn))[valid]
                ix = np.searchsorted(k, lk); ix[ix >= len(k)] = 0
                hit = k[ix] == lk
                cnt = np.zeros((len(k), G), np.int64)
                np.add.at(cnt, (ix[hit], lg[hit]), 1)
            for g in range(G):
                x = bits[:, g] & (own != g)
                for ph in (0, 1):
                    y = x & (hz == bool(ph))
                    c = np.bincount(own[y], weights=cnt[y, g], minlength=G)
                    F[ph, g, :G] += c * B["req"]; F[ph, :G, g] += c * B["data"]
                c = np.bincount(own[x], weights=cnt[x, g], minlength=G)
                F[2, g, :G] += c * B["data"]; F[2, :G, g] += c * B["ack"]
                stats["fetch"] += int(cnt[x, g].sum()); stats["grads"] += int(cnt[x, g].sum())
        out.append(F)
    return out, stats


def pooled_flows(a, B, dedup, to_pool=False):
    """Sharded all-to-all as TorchRec / HugeCTR do it for pooled (multi-hot)
    features: a table lives on one GPU (table-wise sharding); a GPU sends the
    bag's indices to the owner, the owner looks up and sums the rows and
    returns ONE pooled vector per bag; one gradient vector per bag goes back.
    dedup: one exchange per distinct bag (= distinct one-hot id) per GPU.
    Needs --batches (one-hot ids) and the cache's sample placement (gpu)."""
    from sample_placement import MH_SIZES
    G = a.gpus
    out, stats = [], dict(fetch=0, grads=0)
    for i in range(a.first, a.first + a.steps):
        b = np.load(a.batches / f"b{i}.npz")
        d = np.load(a.cache / f"b{i}.npz")
        lengths = b["lengths"]
        T, Bn = lengths.shape
        ids = np.full((T, Bn), -1, np.int64); ids[lengths > 0] = b["indices"]
        gpu = d["gpu"].astype(np.int64) if "gpu" in d else np.arange(Bn) // (Bn // G)
        sizes = MH_SIZES if a.bag_sizes else [1] * T
        order = np.argsort(-np.array(sizes), kind="stable")          # biggest bags on different GPUs
        owner = np.empty(T, np.int64); owner[order] = np.arange(T) % G
        F = np.zeros((4, G + 1, G + 1))
        if to_pool:     # near-memory processing: the pool owns every table and does the pooling
            k = d["keys"]
            stats["uniq_rows"] = stats.get("uniq_rows", 0) + len(k)
            stats["lookups"] = stats.get("lookups", 0) + int(d["raw"])
        for t in range(T):
            o = G if to_pool else owner[t]
            for g in range(G):
                if g == o:
                    continue
                x = ids[t, (gpu == g) & (ids[t] >= 0)]
                n = float(len(np.unique(x)) if dedup else len(x))
                F[1, g, o] += n * 8.0 * sizes[t]; F[1, o, g] += n * B["data"]
                F[2, g, o] += n * B["data"]
                stats["fetch"] += int(n); stats["grads"] += int(n)
        out.append(F)
    return out, stats


def hybrid_flows(a, B, dedup, promote):
    """Hybrid: a pool with near-memory pooling, plus GPUs that keep whole bags.
    Uses the ONE-HOT cache: with the MLPerf multi-hot recipe a bag is a fixed
    function of its one-hot id, so a key is a bag of MH_SIZES[table] rows.
    A bag lives in the pool or is held (all its rows, modified) by one GPU.
      held by its only user   pooled and updated locally, no traffic
      in the pool             every user sends the indices, gets one pooled
                              vector, sends one gradient vector (per lookup,
                              or once per GPU with dedup)
      promotion (lookahead)   a GPU that will be the bag's only user for the
                              next `run` uses fetches its rows if that saves
                              traffic: run * lookups > bag size
      recall                  another GPU uses a held bag: the holder writes
                              its rows back, the bag returns to the pool
    promote=False is the pure near-memory pool."""
    from sample_placement import MH_SIZES
    G, S, W = a.gpus, a.steps, a.keep_window
    steps = load(a.cache, a.first, S + W, a.sample)
    allk = np.unique(np.concatenate([s["k"] for s in steps]))
    NR = len(allk)
    msz = np.array(MH_SIZES, np.int64)
    for s in steps:
        s["rid"] = np.searchsorted(allk, s["k"])
        s["bits"] = bits_of(s["m"], G); s["n"] = s["bits"].sum(1)
        s["cnt"] = np.load(a.cache / f"b{s['i']}.npz")["counts"]
        if a.sample > 1:
            s["cnt"] = s["cnt"][(hsh(np.load(a.cache / f"b{s['i']}.npz")["keys"]) % np.uint64(a.sample)) == 0]
    nrun = np.zeros(NR, np.int64); nmask = np.zeros(NR, np.uint8); ns = np.full(NR, 1 << 30)
    for j in range(len(steps) - 1, -1, -1):
        s = steps[j]; r = s["rid"]
        solo = s["n"] == 1
        cont = solo & (nmask[r] == s["m"]) & (ns[r] - j <= W)
        s["run"] = np.where(solo, 1 + np.where(cont, nrun[r], 0), 0)
        nrun[r] = s["run"]; nmask[r] = s["m"]; ns[r] = j
    holder = np.full(NR, -1, np.int64)
    P = G
    out = []
    st = dict(bag_uses=0, local_uses=0, exchanges=0, promoted=0, recalled=0, pool_rows=0, lookups=0, local_lookups=0, resident=0)
    for s in steps[:S]:
        r, bits, n = s["rid"], s["bits"], s["n"]
        m = msz[s["k"] >> a.row_shift]
        k = np.ones(len(r)) if dedup else np.maximum(s["cnt"] / n, 1.0)      # lookups per using GPU
        h = holder[r]
        g1 = bits.argmax(1)
        solo = n == 1
        hit = solo & (h == g1)
        rec = (h >= 0) & ~hit
        F = np.zeros((4, G + 1, G + 1))
        c = np.bincount(h[rec], weights=m[rec], minlength=G)
        F[0, :G, P] += c * B["data"]; F[0, P, :G] += np.bincount(h[rec], minlength=G) * B["ctl"]
        holder[r[rec]] = -1
        pro = ~hit & solo & promote & (s["run"] * k > m)
        c = np.bincount(g1[pro], weights=m[pro], minlength=G)
        F[0, P, :G] += c * B["data"]; F[0, :G, P] += np.bincount(g1[pro], minlength=G) * B["req"]
        holder[r[pro]] = g1[pro]
        ex = ~hit & ~pro
        for g in range(G):
            x = ex & bits[:, g]
            v = float(k[x].sum())
            F[1, g, P] += float((k[x] * m[x]).sum()) * 8.0; F[1, P, g] += v * B["data"]
            F[2, g, P] += v * B["data"]
            st["exchanges"] += v
            st["pool_rows"] += float((k[x] * m[x]).sum())
        st["bag_uses"] += float((k[:, None] * bits).sum()); st["local_uses"] += float(k[hit | pro].sum())
        st["lookups"] += float(((k * m)[:, None] * bits).sum()); st["local_lookups"] += float((k * m)[hit | pro].sum())
        st["promoted"] += int(pro.sum()); st["recalled"] += int(rec.sum())
        held = holder >= 0
        if held.any():
            mm = msz[allk[held] >> a.row_shift]
            st["resident"] = max(st["resident"], int(np.bincount(holder[held], weights=mm, minlength=G).max()))
        out.append(F)
    return out, st


def pool_flows(a, B, mode, keep, phys):
    """unc_agg / comm, optionally with the software keep schedule."""
    G, S = a.gpus, a.steps
    look = a.keep_window if keep else 0
    steps = load(a.cache, a.first, S + look, a.sample)
    allk = np.unique(np.concatenate([s["k"] for s in steps]))
    NR = len(allk)
    prev = np.array([], np.int64)
    for j, s in enumerate(steps):
        s["rid"] = np.searchsorted(allk, s["k"])
        s["hz"] = np.isin(s["rid"], prev)
        s["bits"] = bits_of(s["m"], G)
        s["n"] = s["bits"].sum(1)
        pick = (s["h"] + j) % s["n"]
        s["agg"] = ((np.cumsum(s["bits"], 1) == (pick + 1)[:, None]) & s["bits"]).argmax(1)
        prev = s["rid"]
    nxt = np.zeros(NR, np.uint8); nxt_s = np.full(NR, 1 << 30)
    for j in range(len(steps) - 1, -1, -1):
        s = steps[j]
        near = nxt_s[s["rid"]] - j <= a.keep_window
        s["defer"] = (s["n"] == 1) & (nxt[s["rid"]] == s["m"]) & near & keep
        nxt[s["rid"]] = s["m"]; nxt_s[s["rid"]] = j
    holder = np.full(NR, -1, np.int64)
    out, stats = [], dict(fetch=0, stores=0, kept=0)
    P = G
    for s in steps[:S]:
        bits, n, hz, defer, agg = s["bits"], s["n"], s["hz"], s["defer"], s["agg"]
        h = holder[s["rid"]]
        F = np.zeros((4, G + 1, G + 1))
        for g in range(G):
            mine = bits[:, g]
            skip = mine & (h == g) & defer & keep
            extra = mine & (h == g) & ~defer & keep
            rd = mine & ~skip & ~(extra if phys else False)
            for ph in (0, 1):
                c = float((rd & (hz == bool(ph))).sum())
                F[ph, g, P] += c * B["req"]; F[ph, P, g] += c * B["data"]
            st_extra = extra if phys else np.zeros_like(extra)
            if mode == "comm":
                w1 = (mine & ~defer & ~extra) | st_extra
                w2s = np.zeros_like(mine); nin = 0
            else:
                isagg = mine & (agg == g)
                w1 = (mine & ~isagg & ~extra) | (st_extra & ~isagg)
                ks = isagg & ~defer
                w2s = (ks & ~extra) | (st_extra & ks)
                nin = float((n[ks] - 1).sum())
            c = float(w1.sum())
            F[2, g, P] += c * B["data"]; F[2, P, g] += c * B["ack"]
            c = float(w2s.sum())
            F[3, g, P] += c * B["data"] + nin * B["req"]; F[3, P, g] += c * B["ack"] + nin * B["data"]
            stats["fetch"] += int(rd.sum()); stats["stores"] += int(w1.sum() + w2s.sum()); stats["kept"] += int(skip.sum())
        holder[s["rid"]] = np.where(n == 1, bits.argmax(1), -1)
        out.append(F)
    return out, stats


def mesi_flows(a, B, stale, deltas=True):
    """MESI through the pool with commutative updates. A row is either held
    modified by one GPU (its only reader last time) or lives in the pool."""
    G, S = a.gpus, a.steps
    steps = load(a.cache, a.first, S, a.sample)
    allk = np.unique(np.concatenate([s["k"] for s in steps]))
    NR = len(allk)
    holder = np.full(NR, -1, np.int64)
    last = np.zeros(NR, np.int64)
    stale_at = np.full((G, NR), -1, np.int64) if stale else None
    prev = np.array([], np.int64)
    out = []
    stats = dict(fetch=0, deltas=0, recalls=0, hits=0, stale_hits=0, evict=0, resident=0)
    P = G
    U = None
    for j, s in enumerate(steps):
        rid = np.searchsorted(allk, s["k"])
        hz = np.isin(rid, prev); prev = rid
        bits = bits_of(s["m"], G)
        n = bits.sum(1)
        F = np.zeros((4, G + 1, G + 1))
        # capacity: evict the least recently used modified rows (write-back)
        if a.cache_rows:
            for g in range(G):
                held = np.flatnonzero(holder == g)
                room = a.cache_rows - int(bits[:, g].sum())
                if len(held) > room:
                    ev = held[np.argsort(last[held])[: len(held) - max(room, 0)]]
                    holder[ev] = -1
                    F[0, g, P] += len(ev) * B["data"]; F[0, P, g] += len(ev) * B["ack"]
                    stats["evict"] += len(ev)
        h = holder[rid]
        g1 = bits.argmax(1)
        solo = n == 1
        hit_only = solo & (h == g1)
        fresh = solo & (h < 0)
        recall = (h >= 0) & ~hit_only
        shared = ~hit_only & ~fresh
        hin = np.zeros(len(rid), bool)
        hin[h >= 0] = bits[np.flatnonzero(h >= 0), h[h >= 0]]
        for ph in (0, 1):
            x = hz == bool(ph)
            # recall: control to the holder, data back to the pool
            c = np.bincount(h[recall & x], minlength=G).astype(float)
            F[ph, P, :G] += c * B["ctl"]; F[ph, :G, P] += c * B["data"]
            # the first reader of an unowned shared row is granted E, then downgraded
            y = (h < 0) & (n >= 2) & x
            c = np.bincount(g1[y], minlength=G).astype(float)
            F[ph, P, :G] += c * B["ctl"]; F[ph, :G, P] += c * B["ctl"]
        order = np.cumsum(bits, 1)                      # position of GPU g among the row's readers
        for g in range(G):
            mine = bits[:, g] & ~hit_only
            own_hit = mine & (h == g)
            rd = mine & ~own_hit
            if stale:
                age_ok = stale_at[g, rid] >= max(j - a.stale_age, 0)
                sh = rd & (stale_at[g, rid] >= 0) & age_ok
                rd = rd & ~sh
                stats["stale_hits"] += int(sh.sum())
                stale_at[g, rid[bits[:, g]]] = -1
            for ph in (0, 1):
                c = float((rd & (hz == bool(ph))).sum())
                F[ph, g, P] += c * B["req"]; F[ph, P, g] += c * B["data"]
            if deltas:
                c = float((bits[:, g] & shared).sum())
                F[2, g, P] += c * B["data"]; F[2, P, g] += c * B["ack"]
            else:
                # plain MESI through the pool: the readers of a shared row take it in turn
                # (GPU order); every writer but the first gets the row from the previous
                # one through the pool, the last one keeps it modified
                mine_s = bits[:, g] & shared
                c = float(mine_s.sum())
                nxt_ = float((mine_s & (order[:, g] < n)).sum())    # hands the row on
                prv_ = float((mine_s & (order[:, g] > 1)).sum())    # receives it
                F[2, g, P] += nxt_ * B["data"] + c * B["ctl"]; F[2, P, g] += prv_ * B["data"] + c * B["ctl"]
            stats["fetch"] += int(rd.sum()); stats["deltas"] += int(c)
            stats["hits"] += int((bits[:, g] & hit_only).sum() + own_hit.sum())
        if stale:
            z = np.flatnonzero(recall & ~hin)
            stale_at[h[z], rid[z]] = j
        stats["recalls"] += int(recall.sum())
        holder[rid[fresh]] = g1[fresh]
        holder[rid[shared]] = -1 if deltas else (G - 1 - bits[shared][:, ::-1].argmax(1))
        last[rid] = j
        stats["resident"] = max(stats["resident"], int(np.bincount(holder[holder >= 0], minlength=G).max()))
        out.append(F)
    return out, stats


def byte_model(a, fabric):
    if a.bytes == "gem5":
        return dict(req=64.0, data=576.0, ack=64.0, ctl=64.0)
    L = a.row_bytes / 64.0
    if fabric == "shard":
        return dict(req=8.0, data=a.row_bytes * (1 + a.nv_overhead), ack=0.0, ctl=8.0)
    if a.pool_bulk:     # host DRAM behind the GPUs' PCIe links: bulk DMA, no per-line messages
        return dict(req=8.0, data=a.row_bytes * (1 + a.pci_overhead), ack=0.0, ctl=8.0)
    return dict(req=L * a.cxl_req, data=L * a.cxl_data, ack=L * a.cxl_ack, ctl=L * a.cxl_req)


def run(a, design):
    phys = a.bytes == "phys"
    if design.startswith("hyb"):
        flows, st = hybrid_flows(a, byte_model(a, "pool"), "dedup" in design, "nopromo" not in design)
        fab = Fabric(a.gpus, a.gpu_bw, a.pool_bw, a.lat_us)
    elif design.startswith("nmp"):
        flows, st = pooled_flows(a, byte_model(a, "pool"), design.endswith("_dedup"), to_pool=True)
        fab = Fabric(a.gpus, a.gpu_bw, a.pool_bw, a.lat_us)
    elif design.startswith("pooled"):
        flows, st = pooled_flows(a, byte_model(a, "shard"), design.endswith("_dedup"))
        fab = Fabric(a.gpus, a.nv_bw, 1.0, a.lat_us)
    elif design.startswith("shard"):
        flows, st = shard_flows(a, byte_model(a, "shard"), design)
        fab = Fabric(a.gpus, a.nv_bw, 1.0, a.lat_us)
    else:
        B = byte_model(a, "pool")
        if design.startswith("mesi"):
            flows, st = mesi_flows(a, B, a.stale, deltas=design != "mesi_pm")
        else:
            flows, st = pool_flows(a, B, design.split("_keep")[0], design.endswith("_keep"), phys)
        fab = Fabric(a.gpus, a.gpu_bw, a.pool_bw, a.lat_us)
    T = np.array([[fab.time(F[p]) for p in range(4)] for F in flows])[a.skip:]
    ph = T.mean(0)
    gb = np.mean([F.sum() for F in flows[a.skip:]]) / 1e6
    n = len(flows)
    return dict(design=design, rp=ph[0], rh=ph[1], w1=ph[2], w2=ph[3], step=ph.sum(), crit=ph[1:].sum(),
                MB=gb, **{k: (v / n if k != "resident" else v) for k, v in st.items()})


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--batches", type=Path)
    ap.add_argument("--designs", default="shard,shard_mig,unc_agg,comm_keep,mesi_comm")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--skip", type=int, default=5, help="steps left out of the average (warm-up)")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--row-shift", type=int, default=26)
    ap.add_argument("--keep-window", type=int, default=16)
    ap.add_argument("--bytes", choices=["gem5", "phys"], default="phys")
    ap.add_argument("--row-bytes", type=float, default=512)
    ap.add_argument("--gpu-bw", type=float, default=38.4, help="GB/s per direction of a GPU's pool-fabric port")
    ap.add_argument("--pool-bw", type=float, default=307.2, help="GB/s per direction of the pool's links, total")
    ap.add_argument("--nv-bw", type=float, default=38.4, help="GB/s per direction per GPU of the GPU-to-GPU fabric")
    ap.add_argument("--nv-overhead", type=float, default=0.0)
    ap.add_argument("--cxl-req", type=float, default=8.0, help="wire bytes of a request per 64 B line")
    ap.add_argument("--cxl-data", type=float, default=72.0, help="wire bytes of a data message per 64 B line")
    ap.add_argument("--cxl-ack", type=float, default=8.0)
    ap.add_argument("--bag-sizes", action="store_true", help="pooled designs: MLPerf multi-hot bag sizes (index bytes)")
    ap.add_argument("--pool-bulk", action="store_true",
                    help="pool designs move rows as bulk transfers (host-DRAM hybrid over PCIe): "
                         "8 B index, row payload x (1 + --pci-overhead), no per-row ack")
    ap.add_argument("--pci-overhead", type=float, default=0.10)
    ap.add_argument("--lat-us", type=float, default=0.0, help="added per non-empty phase")
    ap.add_argument("--cache-rows", type=int, default=0, help="mesi: rows a GPU can hold (0 = unlimited)")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--stale-age", type=int, default=4)
    return ap


def main():
    a = parser().parse_args()
    cols = ["rp", "rh", "w1", "w2", "step", "crit", "MB"]
    print("design," + ",".join(cols) + ",stats")
    for d in a.designs.split(","):
        r = run(a, d)
        extra = " ".join(f"{k}={v:.0f}" for k, v in r.items() if k not in cols and k != "design")
        print(f"{d}," + ",".join(f"{r[c]:.2f}" for c in cols) + "," + extra, flush=True)


if __name__ == "__main__":
    main()
