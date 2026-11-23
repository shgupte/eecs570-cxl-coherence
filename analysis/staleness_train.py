#!/usr/bin/env python3
"""Does stale embedding reads hurt DLRM accuracy? Real training on Criteo 1TB.

Synchronous training makes step s+1 wait for step s's embedding update. With
--delay d, the lookup of step s sees the tables without the last d updates
(the update of step s is applied after the lookup of step s+d). d=0 is exact
synchronous training; d=1 is what a pipelined pool needs to prefetch every
row during the previous step's compute. The dense MLPs are always in sync.

  prep : parquet parts -> one npz (label, 13 dense, 26 categorical ids)
  train: small DLRM (26 tables x --rows rows x --dim, dot interaction), plain
         SGD, global batch 65536; prints log loss and AUC on held-out samples.
"""
import argparse, collections, time
from pathlib import Path
import numpy as np


def prep(a):
    import pyarrow.parquet as pq
    parts = sorted(a.day_dir.glob("*.parquet"))[: a.parts]
    lab, den, cat = [], [], []
    for p in parts:
        t = pq.read_table(p)
        lab.append(t.column("label").to_numpy().astype(np.uint8))
        d = np.stack([t.column(f"integer_feature_{i}").fill_null(0).to_numpy().astype(np.float32)
                      for i in range(1, 14)], 1)
        den.append(np.log1p(np.maximum(d, 0)).astype(np.float16))
        cs = []
        for i in range(1, 27):
            c = t.column(f"categorical_feature_{i}").combine_chunks().dictionary_encode()
            dic = np.array([int(v, 16) if v is not None else 0 for v in c.dictionary.to_pylist()], np.int64)
            ix = c.indices.fill_null(0).to_numpy()
            v = dic[ix] if len(dic) else np.zeros(len(ix), np.int64)
            v[np.asarray(c.is_null())] = 0
            cs.append((v % a.rows).astype(np.int32))
        cat.append(np.stack(cs, 1))
        print(p.name, len(lab[-1]), flush=True)
    np.savez(a.out, label=np.concatenate(lab), dense=np.concatenate(den), cat=np.concatenate(cat))


def auc(y, p):
    o = np.argsort(p, kind="stable")
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def train(a):
    import torch
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    d = np.load(a.data)
    y, den, cat = d["label"], d["dense"], d["cat"]
    T, R, D, B = 26, a.rows, a.dim, a.batch
    n_eval = a.eval_samples
    n_train = (len(y) - n_eval) // B * B
    off = torch.arange(T) * R
    emb = torch.nn.Parameter(torch.empty(T * R, D).uniform_(-0.05, 0.05))
    mlp = lambda sizes: torch.nn.Sequential(*[m for i in range(len(sizes) - 1) for m in
                                              (torch.nn.Linear(sizes[i], sizes[i + 1]), torch.nn.ReLU())][:-1])
    bot, top = mlp([13, 64, D]), mlp([D + (T + 1) * T // 2, 128, 64, 1])
    iu = torch.triu_indices(T + 1, T + 1, 1)
    dense_opt = torch.optim.SGD(list(bot.parameters()) + list(top.parameters()), lr=a.lr_dense)

    def forward(lo, hi):
        x = bot(torch.from_numpy(den[lo:hi].astype(np.float32)))
        ids = torch.from_numpy(cat[lo:hi].astype(np.int64)) + off
        e = torch.nn.functional.embedding(ids, emb, sparse=True)
        z = torch.cat([x[:, None, :], e], 1)
        inter = torch.bmm(z, z.transpose(1, 2))[:, iu[0], iu[1]]
        return top(torch.cat([x, inter], 1)).squeeze(1)

    pending = collections.deque()
    t0 = time.time()
    for s in range(n_train // B):
        lo = s * B
        out = forward(lo, lo + B)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, torch.from_numpy(y[lo:lo + B].astype(np.float32)))
        dense_opt.zero_grad(); emb.grad = None
        loss.backward()
        dense_opt.step()
        g = emb.grad.coalesce()
        pending.append((g.indices()[0], g.values() * (a.lr_emb * B)))   # per-sample-sum scaling
        if len(pending) > a.delay:
            ix, v = pending.popleft()
            with torch.no_grad():
                emb.index_add_(0, ix, -v)
        if s % 50 == 0:
            print(f"step {s} loss {loss.item():.5f} {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        while pending:
            ix, v = pending.popleft(); emb.index_add_(0, ix, -v)
        ps = []
        for lo in range(n_train, n_train + n_eval, B):
            ps.append(torch.sigmoid(forward(lo, min(lo + B, n_train + n_eval))).numpy())
    p = np.clip(np.concatenate(ps), 1e-7, 1 - 1e-7)
    ye = y[n_train:n_train + n_eval].astype(np.float64)
    ll = -(ye * np.log(p) + (1 - ye) * np.log(1 - p)).mean()
    print(f"RESULT delay={a.delay} seed={a.seed} lr_emb={a.lr_emb} lr_dense={a.lr_dense} steps={n_train // B} "
          f"logloss={ll:.5f} auc={auc(ye, p):.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--day-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parts", type=int, default=40)
    p.add_argument("--rows", type=int, default=1 << 20)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--delay", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rows", type=int, default=1 << 20)
    t.add_argument("--dim", type=int, default=16)
    t.add_argument("--batch", type=int, default=65536)
    t.add_argument("--eval-samples", type=int, default=2_000_000)
    t.add_argument("--lr-emb", type=float, default=0.1)
    t.add_argument("--lr-dense", type=float, default=0.1)
    t.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else train(a)


if __name__ == "__main__":
    main()
