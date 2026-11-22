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

Writes per-step caches (keys, counts, mask, rowmax, raw, gpu) readable by
step_traffic.py and linkmodel.py. One-hot batch inputs only (Criteo); pass
--multihot to expand each one-hot id into the MLPerf DLRM-DCNv2 multi-hot bag.
"""
import argparse
from pathlib import Path
import numpy as np

from batch_io import (
    LookupBatch,
    compute_rowmax,
    iter_batches,
    load_onehot_batch,
    pack_onehot,
    save_step_cache,
    scan_table_lengths,
)
from multihot import (
    MH_ROWS,
    MH_SIZES,
    MLPERF_DLRM_V2,
    MultiHotSpec,
    expand_multihot,
    multihot,
)
from placement import (
    BITS,
    VALID_POLICIES,
    CBackend,
    PlacementConfig,
    PlacementResult,
    PythonBackend,
    SamplePlacer,
    SoleUpdaterTracker,
    assign,
    assign_c,
    load_c,
)

__all__ = [
    "BITS",
    "MH_SIZES",
    "MH_ROWS",
    "MLPERF_DLRM_V2",
    "MultiHotSpec",
    "LookupBatch",
    "PlacementConfig",
    "PlacementResult",
    "PythonBackend",
    "CBackend",
    "SoleUpdaterTracker",
    "SamplePlacer",
    "assign",
    "assign_c",
    "load_c",
    "multihot",
    "expand_multihot",
    "pack_onehot",
    "load_onehot_batch",
    "iter_batches",
    "scan_table_lengths",
    "compute_rowmax",
    "save_step_cache",
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", choices=list(VALID_POLICIES), required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--cmax", type=int, default=64, help="rows read more often than this in a step get weight 0")
    ap.add_argument("--lam", type=float, default=1.0, help="load-balance penalty")
    ap.add_argument("--row-shift", type=int, default=26)
    ap.add_argument("--fast", action="store_true", help="C inner loop (placer.c); prints its time per step")
    ap.add_argument("--owner", type=float, default=0.0,
                    help="bonus per row whose static owner (row-wise blocks, as in "
                         "step_traffic.py) is the GPU: placement for a sharded all-to-all system")
    ap.add_argument("--multihot", action="store_true",
                    help="expand every one-hot id to the MLPerf DLRM-DCNv2 multi-hot bag (214 lookups per sample)")
    ap.add_argument("--lag", type=int, default=1,
                    help="affinity: the bonus uses the sole-updater table as of --lag steps earlier "
                         "(a placer pipelined over --lag cores cannot see the steps still in flight)")
    return ap


def run(a: argparse.Namespace) -> None:
    tlen = None
    if a.owner:
        tlen = MLPERF_DLRM_V2.table_lengths if a.multihot else scan_table_lengths(a.batches, a.steps)

    cfg = PlacementConfig(
        policy=a.policy,
        gpus=a.gpus,
        passes=a.passes,
        beta=a.beta,
        cmax=a.cmax,
        lam=a.lam,
        lag=a.lag,
        owner_weight=a.owner,
        fast=a.fast,
    )
    placer = SamplePlacer(cfg, tlen=tlen)

    a.out.mkdir(parents=True, exist_ok=True)
    secs: list[float] = []
    last_B = 0

    for i, batch in iter_batches(a.batches, a.steps, row_shift=a.row_shift):
        if a.multihot:
            batch = expand_multihot(batch, MLPERF_DLRM_V2)
        last_B = batch.batch_size
        res = placer.place(batch)
        if res.elapsed_sec is not None:
            secs.append(res.elapsed_sec)
        save_step_cache(
            a.out / f"b{i}.npz",
            keys=res.keys,
            counts=res.counts,
            mask=res.mask,
            gpu=res.gpu,
            raw=res.raw,
            num_tables=batch.num_tables,
            row_shift=batch.row_shift,
        )
        print(
            f"step {i}: U={res.unique_rows} P={res.pairs} single={res.single_frac:.3f}",
            flush=True,
        )

    if secs:
        print(
            f"placer C loop: mean {np.mean(secs) * 1e3:.2f} ms/step, max {np.max(secs) * 1e3:.2f} ms "
            f"(one core, {last_B} samples)"
        )


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
