#!/usr/bin/env python3
"""Build two-accelerator litmus and DLRM sharing traces from trace-v2."""

import argparse
import csv
from pathlib import Path


LINE_BYTES = 64
ROW_BYTES = 512
PHASE_GAP = 2_000_000
# 32 in flight at the observed ~200k-tick latency drains 2048 requests in
# about 13M ticks. 40M keeps each DLRM phase complete before the next starts.
PHASE_DRAIN = 40_000_000
ISSUE_GAP = 1_000
SHARD_SHIFT = 16 * 1024 * 1024


def write_trace(path: Path, requests: list[tuple[str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        for operation, address, tick in requests:
            if address % LINE_BYTES or tick < 0:
                raise ValueError(f"bad request {operation} {address:#x} {tick}")
            stream.write(f"{operation},{address},{LINE_BYTES},{tick}\n")


def row_lines(base: int) -> list[int]:
    return [base + offset for offset in range(0, ROW_BYTES, LINE_BYTES)]


def load_rows(row_map: Path, count: int) -> list[int]:
    rows = []
    with row_map.open(newline="") as stream:
        for record in csv.DictReader(stream):
            rows.append(int(record["address"]))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"needed {count} rows, found {len(rows)}")
    return rows


def phased(lines: list[int], operation: str, start: int) -> list[tuple[str, int, int]]:
    return [
        (operation, address, start + index * ISSUE_GAP)
        for index, address in enumerate(lines)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()

    litmus = 0x40000000
    write_trace(
        args.output_dir / "litmus-accel0.csv",
        [
            ("r", litmus, 0),
            ("w", litmus, 2 * PHASE_GAP),
            ("r", litmus, 3 * PHASE_GAP),
        ],
    )
    write_trace(
        args.output_dir / "litmus-accel1.csv",
        [
            ("r", litmus, PHASE_GAP),
            ("r", litmus, 3 * PHASE_GAP),
        ],
    )

    lines = [
        address
        for row in load_rows(args.row_map, args.rows)
        for address in row_lines(row)
    ]
    write_start = PHASE_DRAIN
    reread_start = 2 * PHASE_DRAIN
    shared_reads = phased(lines, "r", 0)
    shared_rereads = phased(lines, "r", reread_start)
    write_trace(
        args.output_dir / "shared-accel0.csv",
        shared_reads + phased(lines, "w", write_start) + shared_rereads,
    )
    write_trace(
        args.output_dir / "shared-accel1.csv",
        shared_reads + shared_rereads,
    )
    shifted = [address + SHARD_SHIFT for address in lines]
    write_trace(
        args.output_dir / "sharded-accel0.csv",
        shared_reads + phased(lines, "w", write_start) + shared_rereads,
    )
    write_trace(
        args.output_dir / "sharded-accel1.csv",
        phased(shifted, "r", 0) + phased(shifted, "r", reread_start),
    )
    print(f"rows={args.rows} lines={len(lines)} output={args.output_dir}")


if __name__ == "__main__":
    main()
