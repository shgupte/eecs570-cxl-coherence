#!/usr/bin/env python3
"""How fast can host memory serve embedding rows? Tuned C + OpenMP kernels
(rowops.c) on the real per-step row sets: for each of the 8 GPUs' shares of a
Criteo batch, gather its unique rows from a table in host RAM and apply one
gradient per row. Reports GB/s against threads. Dim 128 (512 B rows)."""
import argparse, ctypes, subprocess, time
from pathlib import Path
import numpy as np

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=8_000_000, help="rows per table")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    a = ap.parse_args()
    src = Path(__file__).with_name("rowops.c"); so = src.with_suffix(".so")
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    lib = ctypes.CDLL(str(so))
    P, L, I, F = ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_float
    lib.fill.argtypes = [P, L, I]; lib.gather.argtypes = [P, P, P, L, I, I]
    lib.scatter_add.argtypes = [P, P, P, L, I, F, I]
    T = 26
    nrow = T * a.cap
    tab = np.empty((nrow, D), np.float32)
    nts = [int(x) for x in a.threads.split(",")]
    t0 = time.perf_counter(); lib.fill(tab.ctypes.data, nrow * D, max(nts))
    print(f"table {nrow:,} rows, {tab.nbytes / 2 ** 30:.1f} GiB, filled in {time.perf_counter() - t0:.0f} s", flush=True)
    sets = []
    for s in range(a.steps):
        d = np.load(a.batches / f"b{s}.npz")
        lengths = d["lengths"]
        ids = np.zeros(lengths.shape, np.int64); ids[lengths > 0] = d["indices"]
        ids = (ids % a.cap + (np.arange(T) * a.cap)[:, None]).T
        B = ids.shape[0] // a.gpus
        sets.append([np.unique(ids[g * B:(g + 1) * B]) for g in range(a.gpus)])
    rows = np.mean([sum(len(u) for u in st) for st in sets])
    mx = max(len(u) for st in sets for u in st)
    out = np.empty((mx, D), np.float32); grad = np.full((mx, D), 1e-6, np.float32)
    print(f"rows per step (8 deduplicated shares): {rows:,.0f} = {rows * D * 4 / 1e6:.0f} MB")
    for nt in nts:
        tg = ts = 0.0
        for rep in range(2):                           # first pass warms up
            tg = ts = 0.0
            for st in sets:
                for u in st:
                    t0 = time.perf_counter(); lib.gather(tab.ctypes.data, u.ctypes.data, out.ctypes.data, len(u), D, nt)
                    t1 = time.perf_counter(); lib.scatter_add(tab.ctypes.data, u.ctypes.data, grad.ctypes.data, len(u), D, -0.01, nt)
                    tg += t1 - t0; ts += time.perf_counter() - t1
        gb = rows * D * 4 / 1e9 * len(sets)
        print(f"threads {nt:2d}: gather {gb / tg:6.1f} GB/s ({tg / len(sets) * 1e3:6.2f} ms/step) | "
              f"update {gb / ts:6.1f} GB/s ({ts / len(sets) * 1e3:6.2f} ms/step)", flush=True)


if __name__ == "__main__":
    main()
