#!/usr/bin/env python3
"""Check completed requests and basic coherence events in the bundled litmus."""

import argparse
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=Path)
    args = parser.parse_args()
    stats = {}
    for line in args.stats.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            stats[fields[0]] = float(fields[1])
        except ValueError:
            pass

    expected = {
        "board.cxl_trace_generator0.totalReads": 2,
        "board.cxl_trace_generator0.totalWrites": 1,
        "board.cxl_trace_generator1.totalReads": 2,
        "board.cxl_trace_generator1.totalWrites": 0,
        "board.cxl_trace_generator0.numSuppressed": 0,
        "board.cxl_trace_generator1.numSuppressed": 0,
    }
    failures = [
        f"{key}: expected {value}, got {stats.get(key, 'missing')}"
        for key, value in expected.items()
        if stats.get(key) != value
    ]
    coherence = "board.cache_hierarchy.ruby_system."
    for key in (
        coherence + "L1Cache_Controller.Inv::total",
        coherence + "L2Cache_Controller.L1_UPGRADE",
    ):
        value = stats.get(key, float("nan"))
        if not math.isfinite(value) or value < 1:
            failures.append(f"{key}: expected at least one event, got {value}")
    if failures:
        raise SystemExit("Litmus FAILED:\n" + "\n".join(failures))
    print("Litmus PASSED: 5 completed requests, no suppressed packets, coherence observed.")


if __name__ == "__main__":
    main()
