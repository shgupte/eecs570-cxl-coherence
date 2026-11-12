#!/usr/bin/env python3
"""Measured cross-step embedding reuse over consecutive Meta DLRM batches.

Pass 1 (per batch, cached as npz): for every (table,row) key touched, its
access count and the bitmask of GPUs that read it (contiguous 8-way sample
split, same as dlrm-sim). Pass 2: compare consecutive steps.

Key questions:
  * of the rows a GPU fetched remotely in step N, how many does it read again
    in step N+1?  Under VI every one of those is re-fetched (the owner's
    update invalidated it); under an update-push or bounded-staleness
    protocol they would hit.
  * does the hot set persist?  top-K by access count in N vs N+1.
Ownership: contiguous TorchRec row-wise blocks, table length = max row id + 1
taken over ALL batches (so ownership is stable across steps).
"""
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np

ROW_SHIFT = 24  # Meta: max row id 12.5M < 2**24. Criteo (40M rows) needs 26.


def load(path):
    if str(path).endswith(".npz"):
        d = np.load(path)
        return d["indices"], d["lengths"]
    import torch
    with gzip.open(path, "rb") as f:
        idx, offsets, lengths = torch.load(f, map_location="cpu")
    return idx.numpy(), lengths.numpy()


def per_batch(path, out, num_gpus):
    if out.exists():
        return
    idx, lengths = load(path)
    T, B = lengths.shape
    rows = idx.astype(np.int64)
    assert rows.max() < (1 << ROW_SHIFT)
    table = np.repeat(np.arange(T, dtype=np.int64), lengths.sum(1))
    per_gpu = B // num_gpus
    sample = np.repeat(np.tile(np.arange(B, dtype=np.int32), T), lengths.ravel())
    gpu = (sample // per_gpu).astype(np.int64)
    del sample
    key = (table << ROW_SHIFT) | rows
    del table, rows
    keys, counts = np.unique(key, return_counts=True)
    pair = np.unique(key * num_gpus + gpu)
    del key, gpu
    pk, pg = pair // num_gpus, (pair % num_gpus).astype(np.uint8)
    starts = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
    mask = np.bitwise_or.reduceat((1 << pg.astype(np.uint16)).astype(np.uint16), starts).astype(np.uint8)
    assert np.array_equal(pk[starts], keys)
    rowmax = np.zeros(T, dtype=np.int64)
    np.maximum.at(rowmax, keys >> ROW_SHIFT, keys & ((1 << ROW_SHIFT) - 1))
    np.savez(out, keys=keys, counts=counts.astype(np.int64), mask=mask, rowmax=rowmax,
             raw=np.int64(lengths.sum()))
    print(f"{path.name}: raw={lengths.sum()} unique={len(keys)}", flush=True)


def owner_of(keys, tlen, num_gpus):
    t = keys >> ROW_SHIFT
    r = keys & ((1 << ROW_SHIFT) - 1)
    block = -(-tlen[t] // num_gpus)
    return (r // block).astype(np.uint8)


def popcount8(m):
    return np.unpackbits(m[:, None], axis=1).sum(1)


def remote_mask(keys, mask, tlen, num_gpus):
    own = owner_of(keys, tlen, num_gpus)
    return mask & ~(np.uint8(1) << own).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--hot-k", default="1000,10000,100000,1000000")
    ap.add_argument("--row-bytes", type=int, default=512)
    ap.add_argument("--pattern", default="fbgemm_t856_bs65536_{i}.pt.gz")
    ap.add_argument("--row-shift", type=int, default=24)
    a = ap.parse_args()
    global ROW_SHIFT
    ROW_SHIFT = a.row_shift
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    G = a.num_gpus
    names = [a.data_dir / a.pattern.format(i=i) for i in range(a.batches)]
    caches = [a.cache_dir / f"b{i}.npz" for i in range(a.batches)]
    for n, c in zip(names, caches):
        per_batch(n, c, G)

    tlen = None
    for c in caches:
        rm = np.load(c)["rowmax"]
        tlen = rm if tlen is None else np.maximum(tlen, rm)
    tlen = tlen + 1

    ks = [int(k) for k in a.hot_k.split(",")]
    rows_out, hot_rows = [], []
    prev = None
    for i, c in enumerate(caches):
        d = np.load(c)
        keys, counts, mask = d["keys"], d["counts"], d["mask"]
        rmask = remote_mask(keys, mask, tlen, G)
        nremote = popcount8(rmask)
        rec = {"step": i, "raw": int(d["raw"]), "unique_rows": len(keys),
               "remote_copies": int(nremote.sum()),
               "rows_with_remote_reader": int((nremote > 0).sum()),
               "mean_gpus_per_row": float(popcount8(mask).mean())}
        order = np.argsort(-counts, kind="stable")
        cum = np.cumsum(counts[order]) / counts.sum()
        for k in ks:
            rec[f"top{k}_access_share"] = float(cum[min(k, len(cum)) - 1])
        if prev is not None:
            pk, pc, prm = prev
            common, ia, ib = np.intersect1d(pk, keys, assume_unique=True, return_indices=True)
            rec["rows_reused_from_prev"] = len(common)
            rec["frac_prev_rows_reused"] = len(common) / len(pk)
            # remote copies installed in N that the same GPU reads again in N+1
            kept = popcount8(prm[ia] & rmask[ib])
            rec["remote_copies_reread"] = int(kept.sum())
            rec["frac_remote_copies_reread"] = int(kept.sum()) / max(1, int(popcount8(prm).sum()))
            # bytes: VI refetches each such copy (512B) + inv+ack (128B);
            # an update-push protocol sends one 512B update instead.
            rec["vi_refetch_bytes"] = int(kept.sum()) * (a.row_bytes + 128)
            rec["push_update_bytes"] = int(kept.sum()) * a.row_bytes
            prev_order = np.argsort(-pc, kind="stable")
            for k in ks:
                hp = pk[prev_order[:k]]
                hn = keys[order[:k]]
                rec[f"top{k}_persist"] = len(np.intersect1d(hp, hn, assume_unique=True)) / k
        rows_out.append(rec)
        print(json.dumps(rec), flush=True)
        prev = (keys, counts, rmask)
        if i == 0:
            # hot rows read by >=2 GPUs, for the gem5 replay
            multi = popcount8(mask) >= 2
            sel = order[multi[order]][:4096]
            hot_rows.append(np.stack([keys[sel], counts[sel], mask[sel],
                                      owner_of(keys[sel], tlen, G)], 1))

    # per-step GPU masks of the step-0 hot rows, so the gem5 trace uses real sharing
    hot = hot_rows[0]
    hk = hot[:, 0]
    steps_mask = np.zeros((len(caches), len(hk)), dtype=np.uint8)
    for i, c in enumerate(caches):
        d = np.load(c)
        pos = np.searchsorted(d["keys"], hk)
        pos = np.minimum(pos, len(d["keys"]) - 1)
        hit = d["keys"][pos] == hk
        steps_mask[i, hit] = d["mask"][pos[hit]]
    np.savez(a.out / "hot_rows.npz", keys=hk, counts=hot[:, 1], mask0=hot[:, 2],
             owner=hot[:, 3], steps_mask=steps_mask)
    with open(a.out / "reuse.json", "w") as f:
        json.dump(rows_out, f, indent=1)
    import csv
    cols = sorted({k for r in rows_out for k in r}, key=lambda k: (k != "step", k))
    with open(a.out / "reuse.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    sys.exit(main())
