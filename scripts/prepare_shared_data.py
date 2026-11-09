#!/usr/bin/env python3
"""Publish versioned traces and a checksum-verified DLRM dataset for the team."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path: Path, size: int, sha256: str) -> None:
    if path.stat().st_size != size or digest(path) != sha256:
        raise ValueError(f"Existing or downloaded file differs from expected input: {path}")


def directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o2770)
        path.chmod(0o2770)
    except FileExistsError:
        if not path.is_dir():
            raise


def publish(destination: Path, size: int, sha256: str, write) -> None:
    """Verify before publishing; never replace an existing shared input."""
    if destination.exists():
        verify(destination, size, sha256)
        print(f"Verified existing: {destination}", flush=True)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".preparing-", delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
        verify(temporary, size, sha256)
        temporary.chmod(0o640)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            verify(destination, size, sha256)
        print(f"Published: {destination}", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--traces-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o007)
    repo = Path(__file__).resolve().parents[1]
    directory(args.shared_root)
    traces = args.shared_root / "traces"
    directory(traces)
    for version in ("trace-v2", "two-accel"):
        target = traces / version
        directory(target)
        for source in sorted((repo / "traces" / version).iterdir()):
            if not source.is_file():
                continue

            def copy(stream, source=source):
                with source.open("rb") as original:
                    shutil.copyfileobj(original, stream)

            publish(target / source.name, source.stat().st_size, digest(source), copy)

    if not args.traces_only:
        metadata_path = repo / "configs" / "dlrm-dataset.json"
        metadata = json.loads(metadata_path.read_text())
        directory(args.dataset.parent)

        def download(stream):
            print(f"Downloading {metadata['size_bytes']} bytes from {metadata['url']}", flush=True)
            with urllib.request.urlopen(metadata["url"], timeout=60) as response:
                shutil.copyfileobj(response, stream)

        publish(args.dataset, metadata["size_bytes"], metadata["sha256"], download)
        publish(
            args.dataset.parent / (args.dataset.name + ".source.json"),
            metadata_path.stat().st_size,
            digest(metadata_path),
            lambda stream: stream.write(metadata_path.read_bytes()),
        )
    print("Shared inputs ready. Existing files were verified and preserved.", flush=True)


if __name__ == "__main__":
    main()
