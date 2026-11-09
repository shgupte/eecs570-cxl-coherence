#!/usr/bin/env python3
"""Inspect a Meta DLRM embedding-bag dataset without generating a trace."""

import argparse
import gzip
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
        "element_size": value.element_size(),
        "bytes": value.numel() * value.element_size(),
        "contiguous": value.is_contiguous(),
    }


def object_summary(value: Any, include_children: bool = True) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, (list, tuple)):
        result: dict[str, Any] = {
            "kind": type(value).__name__,
            "length": len(value),
        }
        if include_children:
            result["children"] = [
                object_summary(child, include_children=False) for child in value
            ]
        return result
    if isinstance(value, dict):
        result = {
            "kind": "dict",
            "keys": [str(key) for key in value.keys()],
        }
        if include_children:
            result["children"] = {
                str(key): object_summary(child, include_children=False)
                for key, child in value.items()
            }
        return result
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    started = time.monotonic()
    with gzip.open(args.dataset, "rb") as stream:
        payload = torch.load(stream, map_location="cpu")
    load_seconds = time.monotonic() - started

    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_size_bytes": args.dataset.stat().st_size,
        "torch_version": torch.__version__,
        "load_seconds": load_seconds,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "payload": object_summary(payload),
    }

    if isinstance(payload, (list, tuple)) and len(payload) == 3:
        names = ("indices", "offsets", "lengths")
        report["embedding_bag_fields"] = {
            name: object_summary(value)
            for name, value in zip(names, payload)
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)

    print(f"loaded={args.dataset}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"max_rss_kib={report['max_rss_kib']}")
    print(f"payload_kind={report['payload']['kind']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
