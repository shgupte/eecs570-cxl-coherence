#!/usr/bin/env python3
"""Measured time of one DLRM training step on a real GPU, split into parts.

Replaces two estimates of the link model with measurements:
  compute  forward + backward + dense update of an MLPerf-size DLRM
           (bottom MLP 13-512-256-128, 26 embeddings of dim 128, dot
           interaction, top MLP 1024-1024-512-256-1) for one GPU's share of
           the batch (65,536 / 8 = 8,192 samples) and for the whole batch
  host     how fast host memory serves embedding rows: gather of the unique
           rows from a table in host RAM, copy to the GPU, copy the gradients
           back, scatter-add update; rows/s and GB/s against CPU threads

Sparse indices are the real Criteo batches (b<i>.npz: indices, lengths);
dense features and labels are random (timing only).
  gpu mode   embeddings in GPU memory (tables capped by --gpu-cap rows each)
  host mode  embeddings in host RAM at full cardinality (or --host-cap)
"""
import argparse, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

D, T = 128, 26


class DLRM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mlp = lambda s: torch.nn.Sequential(*[m for i in range(len(s) - 1) for m in
                                              (torch.nn.Linear(s[i], s[i + 1]), torch.nn.ReLU())][:-1])
        self.bot = mlp([13, 512, 256, D])
        self.top = mlp([D + (T + 1) * T // 2, 1024, 1024, 512, 256, 1])
        self.register_buffer("iu", torch.triu_indices(T + 1, T + 1, 1))

    def forward(self, dense, emb):
        x = torch.relu(self.bot(dense))
        z = torch.cat([x[:, None, :], emb], 1)
        inter = torch.bmm(z, z.transpose(1, 2))[:, self.iu[0], self.iu[1]]
        return self.top(torch.cat([x, inter], 1)).squeeze(1)


def load_ids(batches, i, offs, cap):
    """(B, T) global row ids of batch i (missing features -> row 0 of the table)."""
    d = np.load(batches / f"b{i}.npz")
    lengths = d["lengths"]
    ids = np.zeros(lengths.shape, np.int64)
    ids[lengths > 0] = d["indices"]
    if cap:
        ids %= cap
    return torch.from_numpy((ids + offs[:, None]).T.copy())


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="for the table sizes (rowmax)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=8, help="the batch is split over this many GPUs")
    ap.add_argument("--gpu-cap", type=int, default=2_000_000)
    ap.add_argument("--host-cap", type=int, default=0)
    ap.add_argument("--threads", default="1,2,4,8,16")
    ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    dev = torch.device("cuda")
    print("gpu:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cpu threads", torch.get_num_threads())
    tlen = np.max([np.load(f)["rowmax"] for f in sorted(a.cache.glob("b*.npz"))], axis=0) + 1
    print(f"tables: {T}, rows {int(tlen.sum()):,} ({tlen.sum() * D * 4 / 2 ** 30:.1f} GiB at dim {D})")

    def sizes(cap):
        n = np.minimum(tlen, cap) if cap else tlen
        return np.concatenate([[0], np.cumsum(n)[:-1]]), int(n.sum())

    # ---------------- compute and all-in-GPU step ----------------
    offs, nrow = sizes(a.gpu_cap)
    table = torch.empty(nrow, D, device=dev).uniform_(-0.01, 0.01)
    model = DLRM().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"\n== embeddings in GPU memory ({nrow:,} rows, {nrow * D * 4 / 2 ** 30:.1f} GiB)")
    for amp in (False, True):
        for share in (a.gpus, 1):
            acc = np.zeros(3); n = 0
            for s in range(a.steps):
                ids = load_ids(a.batches, s, offs, a.gpu_cap)
                B = ids.shape[0] // share
                ids = ids[:B].to(dev)
                dense = torch.randn(B, 13, device=dev); y = (torch.rand(B, device=dev) < 0.03).float()
                sync(); t0 = time.perf_counter()
                emb = table[ids].requires_grad_()
                sync(); t1 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = lossf(model(dense, emb).float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sync(); t2 = time.perf_counter()
                table.index_add_(0, ids.reshape(-1), emb.grad.reshape(-1, D), alpha=-a.lr)
                sync(); t3 = time.perf_counter()
                if s >= a.warm:
                    acc += [t1 - t0, t2 - t1, t3 - t2]; n += 1
            lk, cp, up = acc / n * 1e3
            print(f"batch {B:6d} amp={int(amp)}: lookup {lk:.3f} ms, compute (fwd+bwd+dense update) {cp:.3f} ms, "
                  f"embedding update {up:.3f} ms, step {lk + cp + up:.3f} ms")
    del table
    torch.cuda.empty_cache()

    # ---------------- host-memory step ----------------
    offs, nrow = sizes(a.host_cap)
    print(f"\n== embeddings in host memory ({nrow:,} rows, {nrow * D * 4 / 2 ** 30:.1f} GiB)", flush=True)
    t0 = time.perf_counter()
    htab = torch.empty(nrow, D)
    for c in range(0, nrow, 1 << 22):               # touch every page
        htab[c:c + (1 << 22)].fill_(0.001)
    print(f"table filled in {time.perf_counter() - t0:.0f} s", flush=True)
    torch.set_num_threads(1)

    def par(fn, n, nt, pool):
        cuts = np.linspace(0, n, nt + 1).astype(int)
        list(pool.map(lambda j: fn(cuts[j], cuts[j + 1]), range(nt)))

    for nt in [int(x) for x in a.threads.split(",")]:
        pool = ThreadPoolExecutor(nt)
        acc = np.zeros(6); n = 0; rows = 0
        for s in range(a.steps):
            ids = load_ids(a.batches, s, offs, a.host_cap)
            B = ids.shape[0] // a.gpus
            dense = torch.randn(B, 13, device=dev); y = (torch.rand(B, device=dev) < 0.03).float()
            t0 = time.perf_counter()
            # the whole batch: one deduplicated row set per GPU (what the host serves per step)
            shares = [torch.unique(ids[g * B:(g + 1) * B], return_inverse=True) for g in range(a.gpus)]
            t1 = time.perf_counter()
            bufs = [torch.empty(len(u), D).pin_memory() for u, _ in shares]
            t1b = time.perf_counter()
            for (u, _), buf in zip(shares, bufs):
                par(lambda lo, hi: torch.index_select(htab, 0, u[lo:hi], out=buf[lo:hi]), len(u), nt, pool)
            t2 = time.perf_counter()
            u, inv = shares[0]
            g = bufs[0].to(dev, non_blocking=False).requires_grad_(); inv = inv.to(dev)
            sync(); t3 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                loss = lossf(model(dense, g[inv]).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sync(); t4 = time.perf_counter()
            grads = [g.grad.to("cpu")] + [torch.zeros(len(uu), D) for uu, _ in shares[1:]]
            t5 = time.perf_counter()
            for (uu, _), gr in zip(shares, grads):
                par(lambda lo, hi: htab.index_add_(0, uu[lo:hi], gr[lo:hi], alpha=-a.lr), len(uu), nt, pool)
            t6 = time.perf_counter()
            if s >= a.warm:
                acc += [t1 - t0, t2 - t1b, t3 - t2, t4 - t3, t5 - t4, t6 - t5]; n += 1
                rows += sum(len(uu) for uu, _ in shares)
        uq, ga, h2d, cp, d2h, sc = acc / n * 1e3
        r = rows / n; gb = r * D * 4 / 1e9
        print(f"threads {nt:2d}: rows/step {r:,.0f} ({gb * 1e3:.0f} MB) | unique {uq:.2f} ms | gather {ga:.2f} ms "
              f"({gb / ga * 1e3:.1f} GB/s) | scatter-add {sc:.2f} ms ({gb / sc * 1e3:.1f} GB/s) | one GPU share: "
              f"to GPU {h2d:.3f} ms, compute {cp:.3f} ms, grads back {d2h:.3f} ms", flush=True)
        pool.shutdown()


if __name__ == "__main__":
    main()
