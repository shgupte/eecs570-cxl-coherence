#!/usr/bin/env python3
"""Run a conventional DLRM communication model.

The default is TorchRec-style pooled row-wise sharding. ``--model row-transfer``
keeps the full-vector reference. ``--model coherent --protocol vi`` uses a
valid/invalid cache of full rows. All three share the sample partition and
the ownership map.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from baseline import simulate
from coherent import simulate_coherent
from config import SimConfig
from meta_trace import load_meta_batch, subset_batch
from metrics import ASSUMPTIONS, export_csv, export_json
from topology import Ownership, infer_table_rows
from torchrec_baseline import simulate_torchrec
from workload import synthetic_batch, training_steps


def _demo_batch():
    """Bags on a 1,000-row table. With 8 GPUs each shard is 125 rows."""
    bags = {0: {0: [10, 10, 300], 1: [300, 400]}}
    return synthetic_batch(bags, num_samples=8, source="demo-torchrec")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Meta .pt, .pt.gz, or .npz access file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/latest"))
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--bytes-per-element", type=int, default=4)
    parser.add_argument("--request-bytes", type=int, default=8)
    parser.add_argument("--model", choices=("torchrec", "row-transfer", "coherent"), default="torchrec")
    parser.add_argument("--protocol", choices=("vi",), default="vi")
    parser.add_argument(
        "--cache-bytes",
        default="unlimited",
        help="remote-cache capacity in bytes, or unlimited",
    )
    parser.add_argument("--invalidation-bytes", type=int, default=64)
    parser.add_argument("--ack-bytes", type=int, default=64)
    parser.add_argument("--ownership", choices=("contiguous", "modulo"), default="contiguous")
    parser.add_argument("--tables", type=int, help="use the first N table ids in file order")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--samples", type=int, help="contiguous sample count; default is the full batch")
    parser.add_argument("--repeats", type=int, default=1, help="replay the same batch; labeled synthetic")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--demo", action="store_true", help="run the hand-worked 8-GPU example")
    parser.add_argument("--track-versions", action="store_true")
    parser.add_argument("--collect-rows", action="store_true")
    return parser.parse_args(argv)


def _load_workload(args: argparse.Namespace):
    if args.demo:
        if args.dataset is not None:
            raise SystemExit("--demo does not take --dataset")
        return _demo_batch(), "demo-example", False
    if args.dataset is None:
        raise SystemExit("pass --dataset or --demo")
    batch = load_meta_batch(args.dataset)
    subset = False
    table_ids = None
    if args.tables is not None:
        if args.tables < 1 or args.tables > batch.num_tables:
            raise SystemExit(f"--tables must be in 1..{batch.num_tables}")
        table_ids = batch.table_ids[: args.tables]
        subset = args.tables != batch.num_tables
    if args.sample_start != 0 or args.samples is not None:
        subset = True
    batch = subset_batch(
        batch,
        table_ids=table_ids,
        sample_start=args.sample_start,
        sample_count=args.samples,
    )
    origin = "meta-batch-subset" if subset else "meta-batch"
    return batch, origin, subset


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cache_bytes == "unlimited":
        cache_bytes = None
    else:
        cache_bytes = int(args.cache_bytes)
    config = SimConfig(
        num_gpus=args.num_gpus,
        embedding_dim=args.embedding_dim,
        bytes_per_element=args.bytes_per_element,
        request_bytes=args.request_bytes,
        track_versions=args.track_versions,
        collect_rows=args.collect_rows,
        protocol=args.protocol,
        cache_bytes=cache_bytes,
        invalidation_bytes=args.invalidation_bytes,
        acknowledgement_bytes=args.ack_bytes,
    )
    print("loading", flush=True)
    batch, origin, subset = _load_workload(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure_path = args.output_dir / "structure.json"
    structure_path.write_text(json.dumps(batch.describe(), indent=2) + "\n")
    print(
        f"tables={batch.num_tables} samples={batch.batch_size} "
        f"raw_accesses={batch.raw_accesses} synthetic={batch.synthetic}"
    )
    for note in batch.notes:
        print(f"note: {note}")
    if args.describe_only:
        print(f"wrote {structure_path}")
        return 0

    print("simulating", flush=True)
    steps = training_steps(batch, config.num_gpus, repeats=args.repeats, origin=origin)
    if args.repeats > 1:
        print("cross-step replay is synthetic-repeat, not measured production locality")
    if args.ownership == "contiguous":
        if batch.source == "demo-torchrec":
            table_rows = {0: 1000}
            print("note: demo table length is 1000, not the observed maximum row id")
        else:
            table_rows = infer_table_rows(batch)
        ownership = Ownership.contiguous(config.num_gpus, table_rows)
    else:
        table_rows = {}
        ownership = Ownership.modulo(config.num_gpus)
    if args.model == "torchrec":
        results, versions = simulate_torchrec(steps, config, ownership)
    elif args.model == "coherent":
        results, versions, _cache = simulate_coherent(steps, config, ownership)
    else:
        results, versions = simulate(steps, config, ownership)
    workload = {
        "source": batch.source,
        "origin": steps[0].origin,
        "cross_step_locality": steps[0].cross_step_locality,
        "subset": subset,
        "num_tables": batch.num_tables,
        "batch_size": batch.batch_size,
        "sample_start": int(batch.sample_ids[0]),
        "sample_end": int(batch.sample_ids[-1]) + 1,
        "ranges": [
            {
                "gpu_id": item.gpu_id,
                "sample_start": item.sample_start,
                "sample_end": item.sample_end,
            }
            for item in steps[0].ranges
        ],
        "notes": list(batch.notes),
        "model": args.model,
        "ownership": args.ownership,
        "table_rows": {str(table_id): rows for table_id, rows in sorted(table_rows.items())},
    }
    export_json(
        args.output_dir / "metrics.json",
        results,
        config,
        workload=workload,
        assumptions=ASSUMPTIONS,
    )
    export_csv(args.output_dir, results)
    last = results[-1]
    print(
        f"model={last.model} ownership={last.ownership_scheme} steps={len(results)} "
        f"forward_bytes={last.forward_bytes} backward_bytes={last.backward_bytes} "
        f"updated_rows={last.updated_rows} "
        f"remote_partials={last.remote_partial_pooled_vectors} "
        f"reduce_scatter_bytes={last.reduce_scatter_bytes} "
        f"cache_hits={last.cache_hits} cache_misses={last.cache_misses} "
        f"invalidations={last.invalidations}"
    )
    if versions is not None:
        print(f"versioned_rows={len(versions)}")
    print(f"wrote {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
