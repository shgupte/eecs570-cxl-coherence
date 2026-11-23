#!/usr/bin/env python3
"""Criteo 1TB parquet parts -> Meta-style batch files (indices, offsets, lengths).

Each sample has 26 categorical features. Feature f becomes table f; the
32-bit hex hash maps to row int(hash, 16) % --max-rows (MLPerf DLRM caps
tables at 40M rows the same way). A missing value is an empty bag. Bags are
table-major, matching dlrm_datasets: bag = table * batch + sample.

Batches are cut from the parts in filename order. Parquet parts of one day
are Spark partitions, so the order inside a day is not guaranteed to be the
original time order; day boundaries are real.
"""
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

T = 26


def to_rows(col, max_rows):
    arr = col.to_numpy(zero_copy_only=False)
    valid = np.array([v is not None for v in arr])
    rows = np.zeros(len(arr), dtype=np.int64)
    rows[valid] = [int(v, 16) % max_rows for v in arr[valid]]
    return rows, valid


def emit(path, rows, valid):
    # rows, valid: (T, B)
    lengths = valid.astype(np.int32)
    indices = rows[valid]                      # table-major because of (T,B) ravel
    offsets = np.r_[0, np.cumsum(lengths.ravel())].astype(np.int64)
    np.savez(path, indices=indices.astype(np.int64), offsets=offsets, lengths=lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=40_000_000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    buf_rows = [[] for _ in range(T)]
    buf_valid = [[] for _ in range(T)]
    have, made = 0, 0
    for part in sorted(a.day_dir.glob("*.parquet")):
        tab = pq.read_table(part, columns=[f"categorical_feature_{i+1}" for i in range(T)])
        for t in range(T):
            r, v = to_rows(tab.column(t), a.max_rows)
            buf_rows[t].append(r)
            buf_valid[t].append(v)
        have += tab.num_rows
        while have >= a.batch and made < a.batches:
            R = np.stack([np.concatenate(x) for x in buf_rows])
            V = np.stack([np.concatenate(x) for x in buf_valid])
            emit(a.out / f"b{made}.npz", R[:, : a.batch], V[:, : a.batch])
            print(f"b{made} from {part.name}", flush=True)
            made += 1
            buf_rows = [[R[t, a.batch:]] for t in range(T)]
            buf_valid = [[V[t, a.batch:]] for t in range(T)]
            have -= a.batch
        if made >= a.batches:
            break
    print(f"made {made} batches in {a.out}")


if __name__ == "__main__":
    main()
