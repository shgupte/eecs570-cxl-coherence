#!/usr/bin/env python3
"""Distribute one slice of Meta embedding lookups across sharded accelerators.

Each sample goes to one accelerator. Each row is owned by one accelerator,
using contiguous blocks of the observed id span. A lookup becomes 64-byte
reads of a 512-byte row in the owner's 2 GB shard. The owner later writes
that row once. Addresses are packed; raw row ids do not fit in a shard.

This is the memory-read interface for configs/sharded_memory.py. It does not
send pooled all-to-all messages, and it does not start gem5.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


NUM_GPUS = 8
SHARD_BYTES = 2 * 1024**3
ROW_BYTES = 512
LINE_BYTES = 64


def even_shard_sizes(num_rows: int, num_gpus: int) -> list[int]:
    """Contiguous row counts. Leading blocks are ceil(num_rows / num_gpus)."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")
    if num_rows == 0:
        return [0] * num_gpus
    block = (num_rows + num_gpus - 1) // num_gpus
    occupied = num_rows // block
    sizes = [block] * occupied
    if num_rows % num_gpus != 0:
        remainder = num_rows - sum(sizes)
        if remainder > 0:
            sizes.append(remainder)
    if len(sizes) > num_gpus:
        raise RuntimeError("row split produced more shards than accelerators")
    sizes.extend([0] * (num_gpus - len(sizes)))
    return sizes


def owner_ranges(num_rows: int, num_gpus: int) -> list[tuple[int, int]]:
    """Half-open [start, end) row-id interval for each accelerator."""
    ranges = []
    cursor = 0
    for size in even_shard_sizes(num_rows, num_gpus):
        ranges.append((cursor, cursor + size))
        cursor += size
    return ranges


def owner_of(row_id: int, ranges: list[tuple[int, int]]) -> int:
    if row_id < 0:
        raise ValueError("row id must be non-negative")
    for gpu, (start, end) in enumerate(ranges):
        if start <= row_id < end:
            return gpu
    raise ValueError(f"row {row_id} is outside the observed table span")


def partition_samples(sample_start: int, sample_count: int, num_gpus: int) -> list[tuple[int, int]]:
    """Contiguous sample ranges. The first remainder accelerators get one extra sample."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    base, remainder = divmod(sample_count, num_gpus)
    ranges = []
    cursor = sample_start
    for gpu in range(num_gpus):
        count = base + (1 if gpu < remainder else 0)
        ranges.append((cursor, cursor + count))
        cursor += count
    return ranges


def sample_gpu(sample_id: int, ranges: list[tuple[int, int]]) -> int:
    for gpu, (start, end) in enumerate(ranges):
        if start <= sample_id < end:
            return gpu
    raise ValueError(f"sample {sample_id} is outside the slice")


def bag_rows(indices, offsets, batch_size: int, table_id: int, sample_id: int) -> list[int]:
    """Row ids for one bag. indices/offsets are sequences; values may be floats."""
    bag = table_id * batch_size + sample_id
    start = int(offsets[bag])
    end = int(offsets[bag + 1])
    return [int(indices[position]) for position in range(start, end)]


def distribute_lookups(
    lookups: list[tuple[int, int, int]],
    *,
    num_gpus: int = NUM_GPUS,
    shard_bytes: int = SHARD_BYTES,
    row_bytes: int = ROW_BYTES,
    line_bytes: int = LINE_BYTES,
    issue_gap: int = 1000,
    phase_gap: int = 100_000,
    max_reads_per_gpu: int | None = None,
) -> dict:
    """Turn (sample_gpu, table_id, row_id) lookups into per-GPU memory requests.

    Lookups stay in the order given. Duplicate row ids are separate reads.
    ``max_reads_per_gpu`` caps row reads, not cache lines.
    """
    if row_bytes % line_bytes:
        raise ValueError("row size must be a multiple of the line size")
    per_table_max: dict[int, int] = {}
    for _gpu, table_id, row_id in lookups:
        if row_id < 0:
            raise ValueError("row id must be non-negative")
        per_table_max[table_id] = max(per_table_max.get(table_id, -1), row_id)
    ranges = {
        table_id: owner_ranges(max_row + 1, num_gpus)
        for table_id, max_row in per_table_max.items()
    }

    owned_rows: dict[tuple[int, int], list[int]] = {}
    seen_owned: set[tuple[int, int, int]] = set()
    assigned: list[tuple[int, int, int, int]] = []
    read_counts = [0] * num_gpus
    truncated = [False] * num_gpus
    for sample_gpu, table_id, row_id in lookups:
        if not 0 <= sample_gpu < num_gpus:
            raise ValueError(f"sample gpu {sample_gpu} is outside 0..{num_gpus - 1}")
        if max_reads_per_gpu is not None and read_counts[sample_gpu] >= max_reads_per_gpu:
            truncated[sample_gpu] = True
            continue
        owner = owner_of(row_id, ranges[table_id])
        key = (table_id, owner, row_id)
        if key not in seen_owned:
            seen_owned.add(key)
            owned_rows.setdefault((table_id, owner), []).append(row_id)
        assigned.append((sample_gpu, table_id, row_id, owner))
        read_counts[sample_gpu] += 1

    table_base: dict[tuple[int, int], int] = {}
    dense: dict[tuple[int, int, int], int] = {}
    for owner in range(num_gpus):
        cursor = 0
        for table_id in sorted(per_table_max):
            rows = owned_rows.get((table_id, owner), [])
            table_base[(table_id, owner)] = cursor
            for index, row_id in enumerate(rows):
                dense[(table_id, owner, row_id)] = index
            cursor += len(rows) * row_bytes
        if cursor > shard_bytes:
            raise ValueError(
                f"accelerator {owner} needs {cursor} bytes, shard is {shard_bytes}"
            )

    def address(table_id: int, owner: int, row_id: int) -> int:
        packed = dense[(table_id, owner, row_id)]
        return owner * shard_bytes + table_base[(table_id, owner)] + packed * row_bytes

    requests: list[list[tuple[str, int, int, int]]] = [[] for _ in range(num_gpus)]
    ticks = [0] * num_gpus
    local_reads = [0] * num_gpus
    remote_reads = [0] * num_gpus

    def emit(gpu: int, op: str, row_address: int) -> None:
        for offset in range(0, row_bytes, line_bytes):
            requests[gpu].append((op, row_address + offset, line_bytes, ticks[gpu]))
            ticks[gpu] += issue_gap

    for sample_gpu, table_id, row_id, owner in assigned:
        emit(sample_gpu, "r", address(table_id, owner, row_id))
        if owner == sample_gpu:
            local_reads[sample_gpu] += 1
        else:
            remote_reads[sample_gpu] += 1

    read_done = max(ticks) if any(ticks) else 0
    write_at = read_done + phase_gap
    writes = [0] * num_gpus
    for owner in range(num_gpus):
        owned = [
            (table_id, row_id)
            for table_id in sorted(per_table_max)
            for row_id in owned_rows.get((table_id, owner), [])
        ]
        if not owned:
            continue
        ticks[owner] = write_at
        for table_id, row_id in owned:
            emit(owner, "w", address(table_id, owner, row_id))
            writes[owner] += 1

    return {
        "requests": requests,
        "local_row_reads": local_reads,
        "remote_row_reads": remote_reads,
        "row_writes": writes,
        "truncated": truncated,
        "read_done_tick": read_done,
        "write_start_tick": write_at if any(writes) else read_done,
        "last_tick": max(ticks) if any(ticks) else 0,
        "table_rows": {table_id: max_row + 1 for table_id, max_row in per_table_max.items()},
    }


def lookups_from_batch(
    indices,
    offsets,
    lengths,
    *,
    sample_start: int,
    sample_count: int,
    table_ids: list[int],
    num_gpus: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Expand bags in table, then sample, then index order."""
    if getattr(lengths, "ndim", None) != 2:
        raise ValueError("lengths must be a 2-D (tables, samples) array")
    num_tables, batch_size = int(lengths.shape[0]), int(lengths.shape[1])
    if int(offsets.shape[0]) != int(lengths.numel()) + 1:
        raise ValueError("offset count does not match lengths")
    if sample_start < 0 or sample_count < 1 or sample_start + sample_count > batch_size:
        raise ValueError("sample slice is outside the batch")
    ranges = partition_samples(sample_start, sample_count, num_gpus)
    lookups = []
    for table_id in table_ids:
        if table_id < 0 or table_id >= num_tables:
            raise ValueError(f"table {table_id} is outside the batch")
        for sample_id in range(sample_start, sample_start + sample_count):
            rows = bag_rows(indices, offsets, batch_size, table_id, sample_id)
            gpu = sample_gpu(sample_id, ranges)
            lookups.extend((gpu, table_id, row_id) for row_id in rows)
    return lookups, ranges


def write_traces(result: dict, output_dir: Path, *, drain_ticks: int = 20_000_000) -> None:
    """Write one CSV and one traffic-generator config per accelerator.

    Accelerator 0 exits after every accelerator's last request, plus
    ``drain_ticks``. The others idle until that exit. An accelerator with no
    requests keeps a long idle and never exits on its own.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    last = int(result["last_tick"])
    for gpu, requests in enumerate(result["requests"]):
        csv_path = output_dir / f"accel{gpu}.csv"
        with csv_path.open("w") as stream:
            for op, address, size, tick in requests:
                stream.write(f"{op},{address},{size},{tick}\n")
        cfg_path = output_dir / f"accel{gpu}.cfg"
        if not requests:
            cfg_path.write_text(
                "STATE 0 10000000000000 IDLE\n"
                "INIT 0\n"
                "TRANSITION 0 0 1\n"
            )
            continue
        own_last = int(requests[-1][3])
        csv_ref = csv_path.resolve().as_posix()
        if gpu == 0:
            idle = max(1, last - own_last + drain_ticks)
            cfg_path.write_text(
                f"STATE 0 0 CSVTRACE {csv_ref} 0\n"
                f"STATE 1 {idle} IDLE\n"
                "STATE 2 1 EXIT\n"
                "INIT 0\n"
                "TRANSITION 0 1 1\n"
                "TRANSITION 1 2 1\n"
                "TRANSITION 2 2 1\n"
            )
        else:
            cfg_path.write_text(
                f"STATE 0 0 CSVTRACE {csv_ref} 0\n"
                "STATE 1 10000000000000 IDLE\n"
                "INIT 0\n"
                "TRANSITION 0 1 1\n"
                "TRANSITION 1 1 1\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--max-reads-per-gpu", type=int)
    parser.add_argument("--issue-gap", type=int, default=1000)
    parser.add_argument("--phase-gap", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    with gzip.open(args.dataset, "rb") as stream:
        indices, offsets, lengths = torch.load(stream, map_location="cpu")
    num_tables = int(lengths.shape[0])
    if args.tables < 1 or args.tables > num_tables:
        raise SystemExit(f"--tables must be in 1..{num_tables}")
    lookups, ranges = lookups_from_batch(
        indices,
        offsets,
        lengths,
        sample_start=args.sample_start,
        sample_count=args.samples,
        table_ids=list(range(args.tables)),
        num_gpus=args.num_gpus,
    )
    result = distribute_lookups(
        lookups,
        num_gpus=args.num_gpus,
        issue_gap=args.issue_gap,
        phase_gap=args.phase_gap,
        max_reads_per_gpu=args.max_reads_per_gpu,
    )
    write_traces(result, args.output_dir)
    manifest = {
        "dataset": str(args.dataset),
        "model": "sharded-memory-reads",
        "note": (
            "Each accelerator reads the rows in its samples from the owning "
            "shard. The owner writes each distinct row once after every read. "
            "Addresses are packed into the 2 GB shard. This is not pooled all-to-all."
        ),
        "sample_start": args.sample_start,
        "samples": args.samples,
        "tables": list(range(args.tables)),
        "num_gpus": args.num_gpus,
        "shard_bytes": SHARD_BYTES,
        "row_bytes": ROW_BYTES,
        "line_bytes": LINE_BYTES,
        "issue_gap": args.issue_gap,
        "phase_gap": args.phase_gap,
        "table_rows": {str(table_id): rows for table_id, rows in result["table_rows"].items()},
        "read_done_tick": result["read_done_tick"],
        "write_start_tick": result["write_start_tick"],
        "last_tick": result["last_tick"],
        "accelerators": [
            {
                "gpu": gpu,
                "sample_start": ranges[gpu][0],
                "sample_end": ranges[gpu][1],
                "local_row_reads": result["local_row_reads"][gpu],
                "remote_row_reads": result["remote_row_reads"][gpu],
                "row_writes": result["row_writes"][gpu],
                "requests": len(result["requests"][gpu]),
                "truncated": result["truncated"][gpu],
            }
            for gpu in range(args.num_gpus)
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"lookups={len(lookups)} last_tick={result['last_tick']} "
        f"output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
