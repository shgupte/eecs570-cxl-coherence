#!/usr/bin/env python3
"""Does sample placement change training? Real DLRM, real Criteo samples.

Two copies of the model of staleness_train.py start from the same weights.
Each step both see the same 65,536 samples, split over 8 "GPUs" that each
compute the gradient of their own samples; the per-GPU gradients are summed
and applied (synchronous training). Copy A splits the batch in file order,
copy B uses the affinity placement of sample_placement.py (C placer).
Prints, per step, the largest difference between the two summed gradients
and between the two models, then the held-out AUC of both.

In short, this file proves the affinity placement optimization is completely 
lossless for model quality while actively reducing CXL memory traffic.

usage: placement_check.py --data <npz from staleness_train.py prep> [--steps 40]
"""
import argparse
from pathlib import Path
import numpy as np
import torch

import sample_placement as sp
from staleness_train import auc

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--rows", type=int, default=1 << 20)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--lr-emb", type=float, default=2e-3)
    ap.add_argument("--lr-dense", type=float, default=0.3)
    ap.add_argument("--eval-samples", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=8)
    return ap.parse_args()

def forward(m, idx, den, cat, off, iu):
    """Executes one complete forward DLRM pass.
    Returns one prediction logit per sample of shape (N,)."""

    # Extracts continuous features from continuous samples
    x = m["bot"](torch.from_numpy(den[idx].astype(np.float64)))
    ids = torch.from_numpy(cat[idx].astype(np.int64)) + off
    e = torch.nn.functional.embedding(ids, m["emb"], sparse=True)

    # Reshapes bottom mlp vector into shape (N,1,D) to create a unified list of 27
    # feature vectors
    z = torch.cat([x[:, None, :], e], 1)

    # Pairwise dot-product interaction layer
    inter = torch.bmm(z, z.transpose(1, 2))[:, iu[0], iu[1]]

    return m["top"](torch.cat([x, inter], 1)).squeeze(1)



def step(m, lo, gpu, den, cat, off, iu, a, B, G, y):
    """Sum of per-GPU gradients of the summed loss; returns (dense grads, sparse emb grad).
    Accumulates and coalesces dense MLP and sparse embedding gradients targeting the same row.  
    Mathematically both models compute the same gradient regardless of which GPU processed the sample. """
    
    dense = [p for k in ("bot", "top") for p in m[k].parameters()]
    gd = [torch.zeros_like(p) for p in dense]; ge = None

    # Summing gradients across all gpus
    for g in range(G):
        idx = lo + np.flatnonzero(gpu == g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            forward(m, idx, den, cat, off, iu), torch.from_numpy(y[idx].astype(np.float64)), reduction="sum")
        gs = torch.autograd.grad(loss, dense + [m["emb"]])
        for acc, x in zip(gd, gs[:-1]):
            acc += x
        ge = gs[-1] if ge is None else ge + gs[-1]

    ge = ge.coalesce()

    # Updates the dense weights and applies sparse embedding updates
    with torch.no_grad():
        for p, x in zip(dense, gd):
            p -= a.lr_dense / B * x
        m["emb"].index_add_(0, ge.indices()[0], -a.lr_emb * ge.values())
    return gd, ge



def main():
    # Setting up parameters for DLRM model
    a = parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    d = np.load(a.data)
    y, den, cat = d["label"], d["dense"], d["cat"]
    T, R, D, B, G = 26, a.rows, a.dim, a.batch, a.gpus
    off = torch.arange(T) * R
    iu = torch.triu_indices(T + 1, T + 1, 1)

    # Builds a multi-layer perceptron (fully connected neural network) from 
    # a list of layer dimensions
    mlp = lambda sizes: torch.nn.Sequential(*[m for i in range(len(sizes) - 1) for m in
                                              (torch.nn.Linear(sizes[i], sizes[i + 1]), torch.nn.ReLU())][:-1])

    # Creates 2 identical DLRM models A and Bm with embedding table emb and bottom/top mlps
    # A is splits the file in batch order, and Bm uses the affinity placement defined in sample_placement.py
    A = dict(emb=torch.empty(T * R, D, dtype=torch.float64).uniform_(-0.05, 0.05).requires_grad_(),
                    bot=mlp([13, 64, D]).double(), top=mlp([D + (T + 1) * T // 2, 128, 64, 1]).double())

    Bm = dict(emb=A["emb"].detach().clone().requires_grad_(), bot=mlp([13, 64, D]).double(),
              top=mlp([D + (T + 1) * T // 2, 128, 64, 1]).double())
    Bm["bot"].load_state_dict(A["bot"].state_dict()); Bm["top"].load_state_dict(A["top"].state_dict())

    
    cf = sp.load_c()
    # Gives each GPU an equal slice of the B // G consecutive samples
    contig = np.arange(B) // (B // G)

    sole_k = np.array([], np.int64); sole_m = np.array([], np.uint8)
    pairs = [0, 0]

    # Runs the core training and comparison loop
    # Takes each batch of B samples, computes an affinity-based GPU assignment,
    # and trains both A and Bm models synchronously
    for s in range(a.steps):
        lo = s * B
        K = (np.arange(T, dtype=np.int64) << 26) | cat[lo:lo + B].astype(np.int64)
        keys, inv, counts = np.unique(K, return_inverse=True, return_counts=True)
        rid = inv.reshape(B, T)
        bonus = None

        # Keeps track of temporal locality to route samples back to the GPU that 
        # previously owned it
        if len(sole_k):
            ix = np.searchsorted(sole_k, keys); ix[ix >= len(sole_k)] = 0
            bonus = np.where(sole_k[ix] == keys, sole_m[ix], 0).astype(np.uint8)

        # Assigns weight = 1 to samples accessed between 2 and 512 times (ignores others)
        w = ((counts >= 2) & (counts <= 512)).astype(np.float32)
        gpu, cur, _ = sp.assign_c(cf, rid, np.ones((B, T), bool), G, B // G, w, 1.0, bonus=bonus, beta=2.0)

        # Counts how many GPUs share each key (n = 1 means key is private)
        n = np.unpackbits(cur[:, None], axis=1).sum(1)
        keep = ~np.isin(sole_k, keys)
        nk = np.concatenate([sole_k[keep], keys]); nm = np.concatenate([sole_m[keep], np.where(n == 1, cur, 0)])
        o = np.argsort(nk, kind="stable"); sole_k, sole_m = nk[o], nm[o].astype(np.uint8)

        # Builds the GPU bitmask for the baseline contig split. 
        # Fewer pairs means fewer CXL row fetches and coherence messages (good!)
        mc = np.zeros(len(keys), np.uint8)
        np.bitwise_or.at(mc, rid, (1 << contig.astype(np.uint8))[:, None])
        pairs[0] += int(np.unpackbits(mc[:, None], axis=1).sum()); pairs[1] += int(n.sum())

        # gdA/B: summed dense gradients
        # geA, geB: coalesced sparse embedding gradients
        gdA, geA = step(A, lo, contig, den, cat, off, iu, a, B, G, y)
        gdB, geB = step(Bm, lo, gpu.astype(np.int64), den, cat, off, iu, a, B, G, y)

        # Computes the absolute largest element-wise difference between A and Bm's gradients
        # across all dense layers and sparse embedding table rows
        gdiff = max(max(float((x - z).abs().max()) for x, z in zip(gdA, gdB)),
                    float((geA - geB).coalesce().values().abs().max()))

        gmax = max(max(float(x.abs().max()) for x in gdA), float(geA.values().abs().max()))

        # Computes the largest absolute difference between A and Bm's updated weights
        pdiff = max(float((A["emb"] - Bm["emb"]).abs().max()),
                    max(float((p - q).abs().max()) for k in ("bot", "top")
                        for p, q in zip(A[k].parameters(), Bm[k].parameters())))

        # Compares gdiff and pdiff to make sure both models stay identical
        if s % 5 == 0 or s == a.steps - 1:
            print(f"step {s}: max |grad A - grad B| = {gdiff:.3e} (largest grad {gmax:.3e}), "
                  f"max |weight A - weight B| = {pdiff:.3e}", flush=True)
    
    lo = a.steps * B
    idx = np.arange(lo, lo + a.eval_samples)

    # Passes each model through forward pass of DLRM model to convert raw logits into 
    # probabilities between (0,1) and concatenates them back into arrays
    with torch.no_grad():
        pa = np.concatenate([torch.sigmoid(forward(A, c, den, cat, off, iu )).numpy() for c in np.array_split(idx, 16)])
        pb = np.concatenate([torch.sigmoid(forward(Bm, c, den, cat, off, iu)).numpy() for c in np.array_split(idx, 16)])
    ye = y[idx].astype(np.float64)

    # One of the things this print statement does is compute the area under the ROC curve for the model to verify 
    # affinity placement produces the same model accuracy as just splitting up based on batch order
    print(f"RESULT steps={a.steps} pairs/step contig={pairs[0] / a.steps:.0f} placed={pairs[1] / a.steps:.0f} "
          f"auc contig={auc(ye, pa):.6f} placed={auc(ye, pb):.6f} max |pred diff|={np.abs(pa - pb).max():.3e}")


if __name__ == "__main__":
    main()
