#!/usr/bin/env python3
"""Sharing-aware sample placement for synchronous multi-GPU DLRM training.

Which GPU processes which sample of a global batch does not change the
summed gradient, so placement is free. Goal: fewer (GPU,row) pairs per step
(each pair is a row fetch and a gradient return) and more rows whose only
reader is the GPU that already holds them.

Policies (every GPU gets exactly B/G samples):
  contig   samples in file order, B/G per GPU (the baseline everywhere else)
  greedy   one pass; a sample goes to the GPU that already reads most of its
           rows this step, counting only rows read 2..--cmax times in the
           step (hotter rows reach every GPU anyway), minus --lam * load
  refine   greedy, then --passes more passes seeded with the previous pass
  affinity refine + bonus --beta for rows the GPU alone updated last (its
           cached copy is still current), carried across steps

Writes per-step caches (keys, counts, mask, rowmax, raw) readable by
step_traffic.py. One-hot data only (Criteo).
"""
import argparse
from pathlib import Path
import numpy as np

BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little").astype(np.float32)


def assign(rid, valid, G, cap, w, lam, prior=None, bonus=None, beta=0.0):
    """rid (B,T) compact row ids, valid (B,T). prior/bonus: per-row uint8 GPU masks."""
    B = rid.shape[0]
    nrow = int(rid.max()) + 1
    cur = np.zeros(nrow, np.uint8)
    load = np.zeros(G, np.int64)
    out = np.empty(B, np.int8)
    bit = (1 << np.arange(G)).astype(np.uint8)
    for s in range(B):
        r = rid[s][valid[s]]
        wr = w[r][:, None]
        sc = (BITS[cur[r]] * wr).sum(0)[:G]
        if prior is not None:
            sc = sc + 0.5 * (BITS[prior[r]] * wr).sum(0)[:G]
        if bonus is not None:
            sc = sc + beta * BITS[bonus[r]].sum(0)[:G]
        sc = sc - lam * load / cap
        sc[load >= cap] = -1e9
        g = int(sc.argmax())
        out[s] = g
        load[g] += 1
        cur[r] |= bit[g]
    return out, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", choices=["contig", "greedy", "refine", "affinity"], required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--cmax", type=int, default=64, help="rows read more often than this in a step get weight 0")
    ap.add_argument("--lam", type=float, default=1.0, help="load-balance penalty")
    ap.add_argument("--row-shift", type=int, default=26)
    a = ap.parse_args()
    G = a.gpus
    a.out.mkdir(parents=True, exist_ok=True)
    sole_k = np.array([], np.int64); sole_m = np.array([], np.uint8)   # rows a GPU alone updated last
    for i in range(a.steps):
        d = np.load(a.batches / f"b{i}.npz")
        lengths = d["lengths"]
        T, B = lengths.shape
        assert lengths.max() <= 1, "one-hot only"
        valid = (lengths > 0)
        K = np.full((T, B), -1, np.int64)
        K[valid] = (np.repeat(np.arange(T, dtype=np.int64), valid.sum(1)) << a.row_shift) | d["indices"]
        K, valid = K.T.copy(), valid.T.copy()
        keys, inv, counts = np.unique(K[valid], return_inverse=True, return_counts=True)
        rid = np.zeros((B, T), np.int64); rid[valid] = inv
        cap = B // G
        if a.policy == "contig":
            gpu = (np.arange(B) // cap).astype(np.int8)
        else:
            bonus = None
            if a.policy == "affinity" and len(sole_k):
                ix = np.searchsorted(sole_k, keys); ix[ix >= len(sole_k)] = 0
                bonus = np.where(sole_k[ix] == keys, sole_m[ix], 0).astype(np.uint8)
            w = ((counts >= 2) & (counts <= a.cmax)).astype(np.float32)
            gpu, cur = assign(rid, valid, G, cap, w, a.lam, bonus=bonus, beta=a.beta)
            for _ in range(a.passes if a.policy != "greedy" else 0):
                gpu, cur = assign(rid, valid, G, cap, w, a.lam, prior=cur, bonus=bonus, beta=a.beta)
        mask = np.zeros(len(keys), np.uint8)
        np.bitwise_or.at(mask, rid[valid], (1 << gpu.astype(np.uint8))[:, None].repeat(T, 1)[valid])
        rowmax = np.zeros(T, np.int64)
        np.maximum.at(rowmax, keys >> a.row_shift, keys & ((1 << a.row_shift) - 1))
        np.savez(a.out / f"b{i}.npz", keys=keys, counts=counts, mask=mask, rowmax=rowmax,
                 raw=np.int64(valid.sum()), gpu=gpu)
        n = np.unpackbits(mask[:, None], axis=1).sum(1)
        # sole-updater table for the next step
        keep = ~np.isin(sole_k, keys)
        nk = np.concatenate([sole_k[keep], keys])
        nm = np.concatenate([sole_m[keep], np.where(n == 1, mask, 0).astype(np.uint8)])
        o = np.argsort(nk, kind="stable"); sole_k, sole_m = nk[o], nm[o]
        print(f"step {i}: U={len(keys)} P={int(n.sum())} single={float((n == 1).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
