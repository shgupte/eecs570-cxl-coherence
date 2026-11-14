#!/usr/bin/env python3
"""Whole-step embedding traffic of one synchronous training step, per design.

Input: per-step caches (keys, counts, mask, rowmax, raw) written by
reuse_analysis.py or sample_placement.py. Unit: row-sized transfers over a
GPU's link (NIC or CXL port), the bottleneck in both architectures. A
GPU-to-GPU message crosses two GPU links; a pool read or write crosses one.

Per step: L lookups, U unique rows, P (GPU,row) pairs after per-GPU dedup,
P_rem pairs whose GPU is not the row's static owner (TorchRec row-wise block).

Designs
  sq          status quo, sharded all-to-all, one vector per lookup each way:
              4 * L * (G-1)/G   (exact for one-hot data; upper bound otherwise)
  sq_dedup    sharded, owners exchange each unique row once per GPU: 4 * P_rem
  unc_owner   pool, uncacheable (CLAY for training); gradients go through a
              mailbox in the pool to the static owner, which updates the row:
              P reads + 2 * P_rem + owner row reads + U writes
  unc_agg     as above, but one of the row's own readers combines the
              gradients: P + 2 * (P - U) + U
  mesi_rmw    pool with MESI, each reader applies its own update in turn; the
              line moves cache to cache: P + 2 * (P - U) + U
  comm        commutative merge at the pool (COUP-style): each reader sends
              its delta once, the pool adds: P + P
Also reported
  hazard      share of pairs whose row was updated in the previous step (must
              wait for that update; the rest can be prefetched with lookahead)
  retained    share of pairs whose row was last updated by this GPU alone (a
              cached copy is still current: no fetch needed)
"""
import argparse, csv
from pathlib import Path
import numpy as np


def popcount8(m):
    return np.unpackbits(m[:, None], axis=1).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--row-shift", type=int, default=26)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    G = a.gpus
    files = [a.cache / f"b{i}.npz" for i in range(a.steps) if (a.cache / f"b{i}.npz").exists()]
    tlen = np.max([np.load(f)["rowmax"] for f in files], axis=0) + 1
    recs, prev, holder = [], None, {}
    hk = np.array([], np.int64); hg = np.array([], np.int8)   # sole-updater table, sorted keys
    for i, f in enumerate(files):
        d = np.load(f)
        keys, mask, L = d["keys"], d["mask"], int(d["raw"])
        n = popcount8(mask)
        U, P = len(keys), int(n.sum())
        t, r = keys >> a.row_shift, keys & ((1 << a.row_shift) - 1)
        own = (r // -(-tlen[t] // G)).astype(np.uint8)
        owner_reads = (mask >> own) & 1
        P_rem = P - int(owner_reads.sum())
        rec = dict(step=i, L=L, U=U, P=P, gpus_per_row=P / U,
                   sq=4 * L * (G - 1) / G, sq_dedup=4 * P_rem,
                   unc_owner=P + 2 * P_rem + int((owner_reads == 0).sum()) + U,
                   unc_agg=3 * P - U, mesi_rmw=3 * P - U, comm=2 * P)
        if prev is not None:
            rec["hazard"] = float(n[np.isin(keys, prev)].sum() / P)
            ix = np.searchsorted(hk, keys); ix[ix >= len(hk)] = 0
            hit = (hk[ix] == keys) & (hg[ix] >= 0) if len(hk) else np.zeros(U, bool)
            rec["retained"] = float((((mask[hit] >> hg[ix][hit].astype(np.uint8)) & 1)).sum() / P)
        # update sole-updater table: rows touched now -> sole toucher or -1
        sole = np.where(n == 1, np.log2(mask.astype(np.float64)).astype(np.int8), -1).astype(np.int8)
        keep = ~np.isin(hk, keys)
        nk = np.concatenate([hk[keep], keys]); ng = np.concatenate([hg[keep], sole])
        o = np.argsort(nk, kind="stable"); hk, hg = nk[o], ng[o]
        prev = keys
        recs.append(rec)
    cols = list(recs[-1])
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(recs)
    m = {c: np.mean([r[c] for r in recs if c in r]) for c in cols if c != "step"}
    print(f"{a.cache.name}: steps={len(recs)} L={m['L']:.0f} U={m['U']:.0f} P={m['P']:.0f} "
          f"gpus/row={m['gpus_per_row']:.2f} hazard={m['hazard']:.3f} retained={m['retained']:.3f}")
    base = m["unc_agg"]
    for c in ("sq", "sq_dedup", "unc_owner", "unc_agg", "mesi_rmw", "comm"):
        print(f"  {c:10s} {m[c] / 1e3:9.0f}K transfers  {m[c] * 512 / 2**20:8.1f} MiB  x{m[c] / base:.2f} of unc_agg")


if __name__ == "__main__":
    main()
