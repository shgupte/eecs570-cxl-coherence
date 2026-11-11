#!/usr/bin/env python3
"""Print the last complete gem5 statistics dump for the sharded model.

Uses only the Python standard library. Accepts a result directory or stats.txt.
"""

import argparse
import math
from pathlib import Path
import re
import sys


def read_stats(path):
    current = None
    last = None
    for line in path.read_text().splitlines():
        if "Begin Simulation Statistics" in line:
            current = {}
        elif "End Simulation Statistics" in line:
            if current is not None:
                last = current
            current = None
        elif current is not None:
            fields = line.split()
            if len(fields) >= 2:
                try:
                    current[fields[0]] = float(fields[1])
                except ValueError:
                    pass
    if last is None:
        raise ValueError("No complete statistics dump; check the job log for errors.")
    return last


def fmt(value, digits=0):
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def component_ids(stats, pattern):
    result = []
    for key in stats:
        match = re.fullmatch(pattern, key)
        if match:
            result.append(int(match.group(1)))
    return sorted(result)


def render(stats):
    frequency = stats["simFreq"]
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Invalid simFreq in statistics.")
    ns_per_tick = 1e9 / frequency
    sources = component_ids(stats, r"system\.generators(\d+)\.totalReads")
    if not sources:
        raise ValueError("No system.generators<N> statistics found.")

    lines = [
        "Sharded-memory results (last complete statistics dump)",
        f"Simulated duration: {fmt(stats['simTicks'] / frequency * 1e3, 6)} ms",
        f"Simulator wall time: {fmt(stats.get('hostSeconds'), 3)} s",
        "Duration includes scheduled gaps and idle states; it is not training time.",
        "",
        "Accelerator responses and request-response latency",
        " ID    Reads   Writes   Avg read ns  Avg write ns  Retries  Suppressed",
    ]
    read_count = write_count = read_ticks = write_ticks = generated = 0
    suppressed = retries = 0
    for source in sources:
        prefix = f"system.generators{source}."
        reads = stats[prefix + "totalReads"]
        writes = stats[prefix + "totalWrites"]
        read_latency = (stats[prefix + "totalReadLatency"] / reads * ns_per_tick
                        if reads else None)
        write_latency = (stats[prefix + "totalWriteLatency"] / writes * ns_per_tick
                         if writes else None)
        lines.append(
            f"{source:3d} {fmt(reads):>8} {fmt(writes):>8}"
            f" {fmt(read_latency, 3):>13} {fmt(write_latency, 3):>13}"
            f" {fmt(stats[prefix + 'numRetries']):>8}"
            f" {fmt(stats[prefix + 'numSuppressed']):>11}"
        )
        read_count += reads
        write_count += writes
        read_ticks += stats[prefix + "totalReadLatency"]
        write_ticks += stats[prefix + "totalWriteLatency"]
        generated += stats[prefix + "numPackets"]
        suppressed += stats[prefix + "numSuppressed"]
        retries += stats[prefix + "numRetries"]

    completed = read_count + write_count
    lines.extend([
        "",
        f"Responded: {fmt(read_count)} reads + {fmt(write_count)} writes"
        f" = {fmt(completed)} requests",
        f"Generated in-range requests: {fmt(generated)};"
        f" awaiting responses: {fmt(generated - completed)}",
        "Weighted mean read latency: "
        + fmt(read_ticks / read_count * ns_per_tick if read_count else None, 3)
        + " ns",
        "Weighted mean write latency: "
        + fmt(write_ticks / write_count * ns_per_tick if write_count else None, 3)
        + " ns",
        f"Request retry notifications: {fmt(retries)};"
        f" suppressed out-of-range requests: {fmt(suppressed)}",
        "",
        "Memory-owner requests accepted (not response-completion counters)",
        "Owner    Reads   Writes",
    ])
    owners = component_ids(stats, r"system\.mem_ctrls(\d+)\.readReqs")
    for owner in owners:
        prefix = f"system.mem_ctrls{owner}."
        lines.append(f"{owner:5d} {fmt(stats[prefix + 'readReqs']):>8}"
                     f" {fmt(stats[prefix + 'writeReqs']):>8}")

    lines.extend([
        "",
        "Fabric traffic (requests and responses combined)",
        f"Packet traversals: {fmt(stats.get('system.fabric.pktCount::total'))}",
        f"Data payload: {fmt(stats.get('system.fabric.pktSize::total'))} bytes",
        "A remote read normally counts as two packets: request + response.",
        "Payload excludes packet headers, encoding, and protocol overhead.",
        "",
        "Interpretation:",
        "- n/a latency means no responses of that type were observed.",
        "- Latency is reported by the requester across its modeled memory path.",
        "- Means combine local/remote accesses; they do not separate their latency.",
        "- Write responses acknowledge requests; they do not prove DRAM durability.",
        "- Balanced generated/responded counts do not prove the full trace replayed.",
        "- This model currently measures memory/communication, not GPU computation.",
    ])
    if generated != completed:
        lines.append("CHECK: requests were still awaiting responses at the stats dump.")
    if suppressed:
        lines.append("CHECK: some trace addresses were outside the configured memory.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Result directory or stats.txt")
    args = parser.parse_args()
    path = args.results / "stats.txt" if args.results.is_dir() else args.results
    try:
        print(render(read_stats(path)), end="")
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot summarize {path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
