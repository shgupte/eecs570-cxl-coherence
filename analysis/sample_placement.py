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
import argparse, ctypes, subprocess, time
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


def load_c():
    """Compile placer.c next to this file (once) and return its assign()."""
    src = Path(__file__).with_name("placer.c"); so = src.with_suffix(".so")
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    f = ctypes.CDLL(str(so)).assign
    P = ctypes.c_void_p
    f.argtypes = [ctypes.c_int, ctypes.c_int, P, ctypes.c_int, ctypes.c_int, P, ctypes.c_double,
                  P, P, ctypes.c_double, P, ctypes.c_double, P, P]
    return f


def assign_c(f, rid, valid, G, cap, w, lam, prior=None, bonus=None, beta=0.0, owner=None, wown=0.0):
    """Same result as assign(); also returns the seconds spent in the C loop."""
    B, T = rid.shape
    r32 = np.where(valid, rid, -1).astype(np.int32)
    cur = np.zeros(int(rid.max()) + 1, np.uint8); out = np.empty(B, np.int8)
    ptr = lambda x: None if x is None else x.ctypes.data
    w = np.ascontiguousarray(w, np.float32)
    t0 = time.perf_counter()
    f(B, T, r32.ctypes.data, G, cap, w.ctypes.data, lam, ptr(prior), ptr(bonus), beta, ptr(owner), wown,
      cur.ctypes.data, out.ctypes.data)
    return out, cur, time.perf_counter() - t0


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
    ap.add_argument("--fast", action="store_true", help="C inner loop (placer.c); prints its time per step")
    ap.add_argument("--owner", type=float, default=0.0,
                    help="--fast only: bonus per row whose static owner (row-wise blocks, as in "
                         "step_traffic.py) is the GPU: placement for a sharded all-to-all system")
    ap.add_argument("--lag", type=int, default=1,
                    help="affinity: the bonus uses the sole-updater table as of --lag steps earlier "
                         "(a placer pipelined over --lag cores cannot see the steps still in flight)")
    a = ap.parse_args()
    G = a.gpus
    hist = []                                           # sole-updater tables, one per finished step
    cf = load_c() if a.fast else None
    assert a.fast or not a.owner
    tlen = None
    if a.owner:
        T = np.load(a.batches / "b0.npz")["lengths"].shape[0]
        tlen = np.zeros(T, np.int64)
        for i in range(a.steps):
            d = np.load(a.batches / f"b{i}.npz")
            np.maximum.at(tlen, np.repeat(np.arange(T), (d["lengths"] > 0).sum(1)), d["indices"] + 1)
    secs = []
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
            if a.policy == "affinity" and len(hist) >= a.lag and len(hist[-a.lag][0]):
                bk, bm = hist[-a.lag]
                ix = np.searchsorted(bk, keys); ix[ix >= len(bk)] = 0
                bonus = np.where(bk[ix] == keys, bm[ix], 0).astype(np.uint8)
            w = ((counts >= 2) & (counts <= a.cmax)).astype(np.float32)
            if cf:
                own = None
                if a.owner:
                    kt, kr = keys >> a.row_shift, keys & ((1 << a.row_shift) - 1)
                    own = (kr // -(-tlen[kt] // G)).astype(np.int8)
                gpu, cur, dt = assign_c(cf, rid, valid, G, cap, w, a.lam, bonus=bonus, beta=a.beta,
                                        owner=own, wown=a.owner)
                for _ in range(a.passes if a.policy != "greedy" else 0):
                    gpu, cur, d2 = assign_c(cf, rid, valid, G, cap, w, a.lam, prior=cur, bonus=bonus,
                                            beta=a.beta, owner=own, wown=a.owner)
                    dt += d2
                secs.append(dt)
            else:
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
        hist.append((sole_k, sole_m)); hist = hist[-a.lag:]
        print(f"step {i}: U={len(keys)} P={int(n.sum())} single={float((n == 1).mean()):.3f}", flush=True)
    if secs:
        print(f"placer C loop: mean {np.mean(secs) * 1e3:.2f} ms/step, max {np.max(secs) * 1e3:.2f} ms "
              f"(one core, {B} samples)")


if __name__ == "__main__":
    main()
