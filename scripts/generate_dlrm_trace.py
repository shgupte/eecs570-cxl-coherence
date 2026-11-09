#!/usr/bin/env python3
"""Generate a small, reproducible cache-line trace from a Meta DLRM dataset."""

import argparse
import csv
import gzip
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--trace-samples", type=int, default=64)
    parser.add_argument("--selection-samples", type=int, default=256)
    parser.add_argument("--tables", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--element-bytes", type=int, default=4)
    parser.add_argument("--line-bytes", type=int, default=64)
    parser.add_argument("--base-address", type=lambda x: int(x, 0), default=0x40000000)
    parser.add_argument("--table-alignment", type=lambda x: int(x, 0), default=0x200000)
    parser.add_argument("--issue-gap-ticks", type=int, default=1000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.sample_start < 0 or args.trace_samples <= 0:
        raise ValueError("sample range must be positive")
    if args.selection_samples < args.trace_samples:
        raise ValueError("selection-samples must be at least trace-samples")
    if args.tables <= 0:
        raise ValueError("tables must be positive")
    row_bytes = args.embedding_dim * args.element_bytes
    if row_bytes % args.line_bytes:
        raise ValueError("embedding row size must be a multiple of cache-line size")


def write_packet_trace(
    path: Path,
    op: str,
    table_records: list[dict[str, Any]],
    line_bytes: int,
    issue_gap_ticks: int,
    unique_only: bool,
) -> tuple[int, str]:
    tick = 0
    requests = 0
    digest = hashlib.sha256()
    with path.open("w", newline="") as stream:
        for table in table_records:
            compact_rows = (
                range(len(table["raw_row_ids"]))
                if unique_only
                else table["compact_accesses"]
            )
            for compact_row in compact_rows:
                row_address = table["base_address"] + compact_row * table["row_bytes"]
                for line_offset in range(0, table["row_bytes"], line_bytes):
                    line = f"{op},{row_address + line_offset},{line_bytes},{tick}\n"
                    stream.write(line)
                    digest.update(line.encode())
                    requests += 1
                    tick += issue_gap_ticks
    return requests, digest.hexdigest()


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.monotonic()

    with gzip.open(args.dataset, "rb") as stream:
        indices, offsets, lengths = torch.load(stream, map_location="cpu")

    if lengths.ndim != 2:
        raise ValueError(f"expected 2D lengths, got shape {tuple(lengths.shape)}")
    num_tables, batch_size = lengths.shape
    if offsets.numel() != lengths.numel() + 1:
        raise ValueError("offset count does not match flattened lengths")
    if int(offsets[0]) != 0 or int(offsets[-1]) != indices.numel():
        raise ValueError("offset endpoints do not span the indices tensor")
    if args.sample_start + args.selection_samples > batch_size:
        raise ValueError("selection window exceeds dataset batch size")
    if args.tables > num_tables:
        raise ValueError("requested more tables than the dataset contains")

    flat_lengths = lengths.reshape(-1)
    table_stats = []
    selection_end = args.sample_start + args.selection_samples
    trace_end = args.sample_start + args.trace_samples
    for table_id in range(num_tables):
        first_bag = table_id * batch_size + args.sample_start
        last_bag = table_id * batch_size + selection_end
        start = int(offsets[first_bag])
        end = int(offsets[last_bag])
        trace_window_end = int(
            offsets[table_id * batch_size + trace_end]
        )
        expected = int(flat_lengths[first_bag:last_bag].sum())
        if end - start != expected:
            raise ValueError(f"offset/length mismatch for table {table_id}")
        table_indices = indices[start:end]
        if table_indices.numel() and not bool(torch.all(torch.isfinite(table_indices))):
            raise ValueError(f"non-finite index in table {table_id}")
        integer_indices = table_indices.to(torch.int64)
        if table_indices.numel() and not bool(
            torch.all(table_indices == integer_indices.to(table_indices.dtype))
        ):
            raise ValueError(f"non-integral floating-point index in table {table_id}")
        unique_count = int(torch.unique(integer_indices).numel())
        access_count = int(integer_indices.numel())
        table_stats.append(
            {
                "table_id": table_id,
                "selection_accesses": access_count,
                "selection_unique_rows": unique_count,
                "selection_reuse": access_count / unique_count if unique_count else 0.0,
                "trace_window_accesses": trace_window_end - start,
            }
        )

    ranked = sorted(
        (item for item in table_stats if item["trace_window_accesses"] > 0),
        key=lambda item: (
            item["selection_reuse"],
            item["selection_accesses"],
            item["table_id"],
        ),
    )
    if len(ranked) < args.tables:
        raise ValueError("not enough non-empty tables in the selection window")
    rank_positions = [
        round(index * (len(ranked) - 1) / (args.tables - 1))
        if args.tables > 1
        else len(ranked) // 2
        for index in range(args.tables)
    ]
    selected = sorted((ranked[position] for position in rank_positions), key=lambda x: x["table_id"])

    row_bytes = args.embedding_dim * args.element_bytes
    next_base = align_up(args.base_address, args.table_alignment)
    table_records = []
    for selected_stat in selected:
        table_id = selected_stat["table_id"]
        first_bag = table_id * batch_size + args.sample_start
        last_bag = table_id * batch_size + trace_end
        start = int(offsets[first_bag])
        end = int(offsets[last_bag])
        integer_indices = indices[start:end].to(torch.int64)
        raw_rows, inverse = torch.unique(
            integer_indices, sorted=True, return_inverse=True
        )
        raw_row_ids = [int(value) for value in raw_rows.tolist()]
        compact_accesses = [int(value) for value in inverse.tolist()]
        table_bytes = len(raw_row_ids) * row_bytes
        table_records.append(
            {
                **selected_stat,
                "trace_accesses": len(compact_accesses),
                "trace_unique_rows": len(raw_row_ids),
                "trace_reuse": (
                    len(compact_accesses) / len(raw_row_ids)
                    if raw_row_ids
                    else 0.0
                ),
                "base_address": next_base,
                "row_bytes": row_bytes,
                "table_bytes": table_bytes,
                "raw_row_ids": raw_row_ids,
                "compact_accesses": compact_accesses,
            }
        )
        next_base = align_up(next_base + max(table_bytes, 1), args.table_alignment)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reads_path = args.output_dir / "accelerator_reads.csv"
    updates_path = args.output_dir / "unique_row_updates.csv"
    row_map_path = args.output_dir / "row_map.csv"

    read_requests, read_sha = write_packet_trace(
        reads_path,
        "r",
        table_records,
        args.line_bytes,
        args.issue_gap_ticks,
        unique_only=False,
    )
    update_requests, update_sha = write_packet_trace(
        updates_path,
        "w",
        table_records,
        args.line_bytes,
        args.issue_gap_ticks,
        unique_only=True,
    )

    with row_map_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("table_id", "raw_row_id", "compact_row", "address"))
        for table in table_records:
            for compact_row, raw_row_id in enumerate(table["raw_row_ids"]):
                writer.writerow(
                    (
                        table["table_id"],
                        raw_row_id,
                        compact_row,
                        table["base_address"] + compact_row * row_bytes,
                    )
                )

    manifest_tables = []
    for table in table_records:
        manifest_tables.append(
            {
                key: value
                for key, value in table.items()
                if key not in ("raw_row_ids", "compact_accesses")
            }
        )
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "method": {
            "table_selection": (
                "evenly spaced reuse ranks measured over the selection window"
            ),
            "row_mapping": "sorted observed row IDs densely mapped per table",
            "reads": "one full embedding-row read per index occurrence",
            "updates": "one full embedding-row write per unique row",
            "optimizer_model": "fused SGD without optimizer-state traffic",
        },
        "parameters": {
            "sample_start": args.sample_start,
            "trace_samples": args.trace_samples,
            "selection_samples": args.selection_samples,
            "selected_tables": args.tables,
            "embedding_dim": args.embedding_dim,
            "element_bytes": args.element_bytes,
            "row_bytes": row_bytes,
            "line_bytes": args.line_bytes,
            "base_address": args.base_address,
            "table_alignment": args.table_alignment,
            "issue_gap_ticks": args.issue_gap_ticks,
        },
        "dataset_shape": {
            "tables": num_tables,
            "batch_size": batch_size,
            "indices": indices.numel(),
        },
        "tables": manifest_tables,
        "outputs": {
            "accelerator_reads": {
                "path": reads_path.name,
                "requests": read_requests,
                "sha256": read_sha,
            },
            "unique_row_updates": {
                "path": updates_path.name,
                "requests": update_requests,
                "sha256": update_sha,
            },
            "row_map": row_map_path.name,
        },
        "generation": {
            "seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "torch_version": torch.__version__,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"selected_tables={[table['table_id'] for table in table_records]}")
    print(f"read_requests={read_requests}")
    print(f"update_requests={update_requests}")
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
