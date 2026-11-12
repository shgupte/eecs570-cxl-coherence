#!/usr/bin/env python3
"""Distribute one slice of Meta embedding lookups as a row-wise collective.

Each sample goes to one accelerator. Each row is owned by one accelerator.
The sample's accelerator sends the ids to the owner. The owner reads its own
rows locally and sends back one 512-byte pooled result per bag. After every
lookup in the slice, the sample's accelerator sends one 512-byte gradient to
each owner that contributed, and the owner writes each distinct row once.
Those gradient messages happen once per batch, not once per row.

Addresses are packed into the owner's 2 GB shard. This does not start gem5,
and it does not implement a real NCCL reduce-scatter. The pooled result and
the gradient are single transfers of the right size.
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
ID_BYTES = 8


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
    lookups: list[tuple[int, int, int, int]],
    *,
    num_gpus: int = NUM_GPUS,
    shard_bytes: int = SHARD_BYTES,
    row_bytes: int = ROW_BYTES,
    line_bytes: int = LINE_BYTES,
    id_bytes: int = ID_BYTES,
    issue_gap: int = 1000,
    phase_gap: int = 100_000,
    max_reads_per_gpu: int | None = None,
) -> dict:
    """Turn lookups into the production row-wise exchange.

    Each lookup is ``(sample_gpu, sample_id, table_id, row_id)``. One bag is
    the lookups that share sample, table, and order. Duplicate ids are kept.
    A gradient is sent once per bag per remote owner, after every lookup.
    """
    if row_bytes % line_bytes or id_bytes < 1:
        raise ValueError("row size must be a multiple of the line size")
    per_table_max: dict[int, int] = {}
    for sample_gpu, _sample_id, table_id, row_id in lookups:
        if row_id < 0:
            raise ValueError("row id must be non-negative")
        if not 0 <= sample_gpu < num_gpus:
            raise ValueError(f"sample gpu {sample_gpu} is outside 0..{num_gpus - 1}")
        per_table_max[table_id] = max(per_table_max.get(table_id, -1), row_id)
    ranges = {
        table_id: owner_ranges(max_row + 1, num_gpus)
        for table_id, max_row in per_table_max.items()
    }

    bags: list[tuple[tuple[int, int, int], list[tuple[int, int]]]] = []
    bag_index: dict[tuple[int, int, int], int] = {}
    kept = [0] * num_gpus
    truncated = [False] * num_gpus
    for sample_gpu, sample_id, table_id, row_id in lookups:
        if max_reads_per_gpu is not None and kept[sample_gpu] >= max_reads_per_gpu:
            truncated[sample_gpu] = True
            continue
        owner = owner_of(row_id, ranges[table_id])
        key = (sample_gpu, sample_id, table_id)
        if key not in bag_index:
            bag_index[key] = len(bags)
            bags.append((key, []))
        bags[bag_index[key]][1].append((row_id, owner))
        kept[sample_gpu] += 1

    owned_rows: dict[tuple[int, int], list[int]] = {}
    seen_owned: set[tuple[int, int, int]] = set()
    for (_sample_gpu, _sample_id, table_id), rows in bags:
        for row_id, owner in rows:
            key = (table_id, owner, row_id)
            if key not in seen_owned:
                seen_owned.add(key)
                owned_rows.setdefault((table_id, owner), []).append(row_id)

    dense: dict[tuple[int, int, int], int] = {}
    row_bytes_used = [0] * num_gpus
    for owner in range(num_gpus):
        cursor = 0
        for table_id in sorted(per_table_max):
            for index, row_id in enumerate(owned_rows.get((table_id, owner), [])):
                dense[(table_id, owner, row_id)] = index
                cursor += row_bytes
        row_bytes_used[owner] = cursor
        if cursor > shard_bytes:
            raise ValueError(
                f"accelerator {owner} needs {cursor} bytes, shard is {shard_bytes}"
            )

    mailbox = list(row_bytes_used)

    def row_address(table_id: int, owner: int, row_id: int) -> int:
        return owner * shard_bytes + dense[(table_id, owner, row_id)] * row_bytes

    def alloc(shard: int, nbytes: int) -> int:
        start = mailbox[shard]
        aligned = (start + line_bytes - 1) // line_bytes * line_bytes
        mailbox[shard] = aligned + nbytes
        if mailbox[shard] > shard_bytes:
            raise ValueError(f"accelerator {shard} mailbox exceeds the shard")
        return shard * shard_bytes + aligned

    id_sends: list[tuple[int, int, int, int]] = []
    local_reads: list[tuple[int, int, int]] = []
    results: list[tuple[int, int, int]] = []
    gradients: list[tuple[int, int, int]] = []
    for (sample_gpu, _sample_id, table_id), rows in bags:
        by_owner: dict[int, list[int]] = {}
        for row_id, owner in rows:
            by_owner.setdefault(owner, []).append(row_id)
        for owner, row_ids in by_owner.items():
            for row_id in row_ids:
                local_reads.append((owner, table_id, row_id))
            if owner == sample_gpu:
                continue
            slot = alloc(owner, id_bytes * len(row_ids))
            for index in range(len(row_ids)):
                id_sends.append((sample_gpu, owner, slot + index * id_bytes, id_bytes))
            results.append((owner, sample_gpu, alloc(sample_gpu, row_bytes)))
            gradients.append((sample_gpu, owner, alloc(owner, row_bytes)))

    def burst(events: list[tuple[int, str, int, int]]) -> list[list[tuple[str, int, int]]]:
        per_gpu: list[list[tuple[str, int, int]]] = [[] for _ in range(num_gpus)]
        for gpu, op, addr, size in events:
            if size == row_bytes and line_bytes < row_bytes:
                for offset in range(0, row_bytes, line_bytes):
                    per_gpu[gpu].append((op, addr + offset, line_bytes))
            else:
                per_gpu[gpu].append((op, addr, size))
        return per_gpu

    phases = [
        burst([(gpu, "w", addr, size) for gpu, _owner, addr, size in id_sends]),
        burst([
            (owner, "r", row_address(table_id, owner, row_id), row_bytes)
            for owner, table_id, row_id in local_reads
        ]),
        burst([(owner, "w", addr, row_bytes) for owner, _dest, addr in results]),
        burst([(gpu, "w", addr, row_bytes) for gpu, _owner, addr in gradients]),
        burst([
            (owner, "w", row_address(table_id, owner, row_id), row_bytes)
            for owner in range(num_gpus)
            for table_id in sorted(per_table_max)
            for row_id in owned_rows.get((table_id, owner), [])
        ]),
    ]
    requests: list[list[tuple[str, int, int, int]]] = [[] for _ in range(num_gpus)]
    cursors = [0] * num_gpus
    phase_starts = []
    for index, phase in enumerate(phases):
        start = 0 if index == 0 else max(cursors) + phase_gap
        phase_starts.append(start)
        cursors = [start] * num_gpus
        for gpu in range(num_gpus):
            for op, addr, size in phase[gpu]:
                requests[gpu].append((op, addr, size, cursors[gpu]))
                cursors[gpu] += issue_gap
    writes = [0] * num_gpus
    for owner in range(num_gpus):
        writes[owner] = sum(len(owned_rows.get((table_id, owner), [])) for table_id in per_table_max)
    return {
        "requests": requests,
        "id_sends": [sum(1 for gpu, _owner, _addr, _size in id_sends if gpu == index) for index in range(num_gpus)],
        "local_row_reads": [sum(1 for owner, _table, _row in local_reads if owner == index) for index in range(num_gpus)],
        "pooled_results": [sum(1 for owner, _dest, _addr in results if owner == index) for index in range(num_gpus)],
        "gradients": [sum(1 for gpu, _owner, _addr in gradients if gpu == index) for index in range(num_gpus)],
        "row_writes": writes,
        "truncated": truncated,
        "phase_starts": phase_starts,
        "gradient_start_tick": phase_starts[3],
        "update_start_tick": phase_starts[4],
        "last_tick": max(cursors) if any(cursors) else 0,
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
            lookups.extend((gpu, sample_id, table_id, row_id) for row_id in rows)
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
        "model": "sharded-pooled-exchange",
        "note": (
            "Ids are sent to the row owner. The owner reads locally and returns "
            "one pooled result per bag. After all lookups, one pooled gradient "
            "goes back to each contributing owner, then the owner writes each "
            "distinct row once. Addresses are packed. This is not an NCCL collective."
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
        "gradient_start_tick": result["gradient_start_tick"],
        "update_start_tick": result["update_start_tick"],
        "last_tick": result["last_tick"],
        "accelerators": [
            {
                "gpu": gpu,
                "sample_start": ranges[gpu][0],
                "sample_end": ranges[gpu][1],
                "id_sends": result["id_sends"][gpu],
                "local_row_reads": result["local_row_reads"][gpu],
                "pooled_results": result["pooled_results"][gpu],
                "gradients": result["gradients"][gpu],
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
