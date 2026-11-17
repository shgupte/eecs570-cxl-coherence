#!/usr/bin/env python3
"""How much does a serving model lose when it lags the trainer?

One DLRM trains online over a time-ordered Criteo stream (global batch
65,536, plain SGD). Before the trainer learns from batch s, several "serving"
copies predict it: copy k is the trainer's model as it was --lags[k] steps
ago (lag 0 = serving reads the trainer's tables directly). Reports log loss,
normalized entropy and AUC of every copy over the steps after --warm.

Two readings per lag L:
  total   serving copy (model of step s-L) on batch s  vs  lag 0 on batch s
          (includes that the fresh model has also seen L more batches)
  drift   model of step s-L on batch s  vs  the same model on batch s-L
          (only the change in the data; a lower bound on the value of freshness)
"""
import argparse, collections
import numpy as np
import torch

T = 26


def auc(y, p):
    o = np.argsort(p, kind="stable"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lags", default="0,2,8,32,128")
    ap.add_argument("--rows", type=int, default=1 << 20)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr-emb", type=float, default=2e-3)
    ap.add_argument("--lr-dense", type=float, default=0.3)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--steps", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    d = np.load(a.data)
    y, den, cat = d["label"], d["dense"], d["cat"]
    B, R, D = a.batch, a.rows, a.dim
    S = len(y) // B if not a.steps else min(a.steps, len(y) // B)
    lags = [int(x) for x in a.lags.split(",")]
    print(f"samples {len(y):,}, steps {S}, click rate {y.mean():.4f}, lags {lags}", flush=True)
    off = (torch.arange(T, device=dev) * R)
    iu = torch.triu_indices(T + 1, T + 1, 1, device=dev)
    mlp = lambda s: torch.nn.Sequential(*[m for i in range(len(s) - 1) for m in
                                          (torch.nn.Linear(s[i], s[i + 1]), torch.nn.ReLU())][:-1])
    emb = torch.empty(T * R, D, device=dev).uniform_(-0.05, 0.05)
    bot, top = mlp([13, 64, D]).to(dev), mlp([D + (T + 1) * T // 2, 128, 64, 1]).to(dev)
    dense = list(bot.parameters()) + list(top.parameters())

    def forward(e, bt, tp, x, ids):
        h = bt(x)
        z = torch.cat([h[:, None, :], e], 1)
        inter = torch.bmm(z, z.transpose(1, 2))[:, iu[0], iu[1]]
        return tp(torch.cat([h, inter], 1)).squeeze(1)

    import copy
    serve = {L: dict(emb=emb.clone(), bot=copy.deepcopy(bot), top=copy.deepcopy(top)) for L in lags if L}
    queue = collections.deque()                 # (ids, emb delta, dense state) per step
    loss_sum = {L: np.zeros(S) for L in lags}
    preds = {L: [] for L in lags}; labels = []
    for s in range(S):
        sl = slice(s * B, (s + 1) * B)
        x = torch.from_numpy(den[sl].astype(np.float32)).to(dev)
        ids = torch.from_numpy(cat[sl].astype(np.int64)).to(dev) + off
        yy = torch.from_numpy(y[sl].astype(np.float32)).to(dev)
        with torch.no_grad():                   # serving copies predict first
            for L in lags:
                if L == 0:
                    continue
                m = serve[L]
                lg = forward(m["emb"][ids], m["bot"], m["top"], x, ids)
                loss_sum[L][s] = float(torch.nn.functional.binary_cross_entropy_with_logits(lg, yy))
                if s >= a.warm and s % 4 == 0:
                    preds[L].append(torch.sigmoid(lg[::8]).cpu().numpy())
        e = emb[ids].requires_grad_()
        lg = forward(e, bot, top, x, ids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, yy, reduction="sum")
        loss_sum[0][s] = float(loss) / B
        if s >= a.warm and s % 4 == 0:
            preds[0].append(torch.sigmoid(lg[::8]).detach().cpu().numpy()); labels.append(y[sl][::8])
        gs = torch.autograd.grad(loss, dense + [e])
        with torch.no_grad():
            for p, g in zip(dense, gs[:-1]):
                p -= a.lr_dense / B * g
            delta = -a.lr_emb * gs[-1].reshape(-1, D)
            flat = ids.reshape(-1)
            emb.index_add_(0, flat, delta)
            queue.append((flat, delta, [p.detach().clone() for p in dense]))
            for L in serve:                     # copy L now becomes the model after step s-L
                if len(queue) > L:
                    f, dl, ds = queue[len(queue) - 1 - L]
                    m = serve[L]
                    m["emb"].index_add_(0, f, dl)
                    for p, q in zip(list(m["bot"].parameters()) + list(m["top"].parameters()), ds):
                        p.copy_(q)
            while len(queue) > max(lags) + 1:
                queue.popleft()
        if s % 50 == 0:
            print(f"step {s}: train loss {loss_sum[0][s]:.4f}", flush=True)
    yv = np.concatenate(labels); base = -(yv.mean() * np.log(yv.mean()) + (1 - yv.mean()) * np.log(1 - yv.mean()))
    w = slice(a.warm, S)
    f0 = loss_sum[0][w].mean()
    print(f"\nsteps {a.warm}..{S - 1}; one step = {B:,} samples (about half a minute of this stream)")
    print("lag_steps,logloss,norm_entropy,auc,total_loss_increase_%,drift_only_increase_%")
    for L in lags:
        l = loss_sum[L][w].mean()
        same = loss_sum[0][a.warm - L:S - L].mean()      # the lagged model on its own batch
        print(f"{L},{l:.5f},{l / base:.5f},{auc(yv, np.concatenate(preds[L])):.5f},{(l / f0 - 1) * 100:.3f},{(l / same - 1) * 100:.3f}")


if __name__ == "__main__":
    main()
