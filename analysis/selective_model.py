#!/usr/bin/env python3
"""Byte / critical-path model of selective coherence over consecutive batches.

Input: per-batch caches from reuse_analysis.py (keys, counts, GPU masks).
Ownership: contiguous row-wise blocks (same as reuse_analysis / dlrm-sim).
Unit: one (GPU g, row r) remote use per step = g reads r and g != owner(r).

Protocols, per remote use (R = row bytes, 512 by default):
  send      : owner sends the row, g returns a gradient.          fetch R + grad R
  vi        : coherent cache of every row. Every touched row is updated each
              step, so the copy from the last step is always invalid:
              fetch R + grad R + inv/ack (64+64) when g held a copy.
  push-K    : rows in the hot set H_s (top-K by count in step s-1, chosen
              online) are cached; after each update the owner pushes the new
              row to every GPU holding a copy (R per holder per step, used or
              not). A read of a held hot row is a local hit (no fetch on the
              critical path). Cold rows use send. Rows leaving H are dropped.
  stale-K-s: like push-K but the owner pushes only every s steps; in between
              holders read their (bounded-stale) copy.
Gradients always go back to the owner (R per use) in every protocol.
Reported per step: total bytes, critical-path fetch bytes (forward-pass
fetches), push bytes, and wasted push bytes (pushed copies not read before
the next push).
"""
import argparse, json
from pathlib import Path
import numpy as np


def popcount8(m):
    return np.unpackbits(m[:, None], axis=1).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--batches", type=int, required=True)
    ap.add_argument("--row-shift", type=int, default=26)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--row-bytes", type=int, default=512)
    ap.add_argument("--ks", default="1000,10000,100000")
    ap.add_argument("--stale", default="1,2,4")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    G, R, SH = a.num_gpus, a.row_bytes, a.row_shift
    steps = [np.load(a.cache_dir / f"b{i}.npz") for i in range(a.batches)]
    tlen = np.max([s["rowmax"] for s in steps], axis=0) + 1

    def remote(s):
        k, m = s["keys"], s["mask"]
        t, r = k >> SH, k & ((1 << SH) - 1)
        own = (r // -(-tlen[t] // G)).astype(np.uint8)
        return k, m & ~(np.uint8(1) << own).astype(np.uint8), s["counts"]

    data = [remote(s) for s in steps]
    res = []
    # send / vi baselines
    for i in range(1, a.batches):
        k, rm, _ = data[i]
        uses = int(popcount8(rm).sum())
        pk, prm, _ = data[i - 1]
        common, ia, ib = np.intersect1d(pk, k, assume_unique=True, return_indices=True)
        held = int(popcount8(prm[ia] & rm[ib]).sum())
        res.append(dict(step=i, protocol="send", total=2 * R * uses, crit_fetch=R * uses, push=0, wasted=0, uses=uses))
        res.append(dict(step=i, protocol="vi", total=2 * R * uses + 128 * held, crit_fetch=R * uses, push=0, wasted=0, uses=uses))
    for K in [int(x) for x in a.ks.split(",")]:
        for S in [int(x) for x in a.stale.split(",")]:
            holders = {}  # key -> uint8 mask of GPUs holding a copy
            unread = {}   # key -> mask of holders that have not read since last push
            for i in range(1, a.batches):
                pk, _, pc = data[i - 1]
                hot = pk[np.argsort(-pc, kind="stable")[:K]]
                hotset = set(hot.tolist())
                # rows leaving H are dropped
                for key in list(holders):
                    if key not in hotset:
                        del holders[key]
                        unread.pop(key, None)
                k, rm, _ = data[i]
                is_hot = np.isin(k, hot)
                uses_all = popcount8(rm)
                cold_uses = int(uses_all[~is_hot].sum())
                hot_idx = np.flatnonzero(is_hot)
                hit = miss = 0
                for j in hot_idx:
                    key, m = int(k[j]), int(rm[j])
                    if not m:
                        continue
                    h = holders.get(key, 0)
                    hit += bin(m & h).count("1")
                    miss += bin(m & ~h & 0xFF).count("1")
                    holders[key] = h | m
                    unread[key] = unread.get(key, 0) & ~m
                # end of step i: owner updates touched hot rows; push every S steps
                push = wasted = 0
                if i % S == 0:
                    for key, h in holders.items():
                        n = bin(h).count("1")
                        push += R * n
                        wasted += R * bin(unread.get(key, 0)).count("1")
                        unread[key] = h
                uses = int(uses_all.sum())
                crit = R * (cold_uses + miss)
                total = crit + R * uses + push  # fetches + grads + pushes
                res.append(dict(step=i, protocol=f"push-K{K}-s{S}", total=total, crit_fetch=crit,
                                push=push, wasted=wasted, uses=uses, hot_hits=hit, hot_misses=miss))
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "selective_steps.json", "w") as f:
        json.dump(res, f)
    # summary: mean over steps 2.. (step 1 is cold for every cache)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in res:
        if r["step"] >= 2:
            agg[r["protocol"]].append(r)
    lines = ["protocol,total_MB,crit_fetch_MB,push_MB,wasted_push_MB,hot_hit_rate"]
    for p, rs in agg.items():
        m = lambda f: np.mean([r.get(f, 0) for r in rs]) / 1e6
        hh = sum(r.get("hot_hits", 0) for r in rs)
        hm = sum(r.get("hot_misses", 0) for r in rs)
        rate = hh / (hh + hm) if hh + hm else float("nan")
        lines.append(f"{p},{m('total'):.2f},{m('crit_fetch'):.2f},{m('push'):.2f},{m('wasted'):.2f},{rate:.3f}")
    (a.out / "selective_summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
