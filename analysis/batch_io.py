#!/usr/bin/env python3
"""Batch loading, one-hot key packing, and per-step cache I/O for DLRM placement.

Input batches follow the table-major format emitted by criteo_to_batches.py:
  lengths: (T, B) int32, 0 or 1 (one-hot only)
  indices: (nnz,) int64, row ids in table-major order

Output per-step caches follow the schema consumed by step_traffic.py and
linkmodel.py:
  keys, counts, mask, rowmax, raw, gpu
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np


@dataclass(frozen=True)
class LookupBatch:
    """Sample-major embedding lookup batch.

    keys:       (B, Tc) int64 packed as (table << row_shift) | row_id (-1 if invalid)
    valid:      (B, Tc) bool mask of present lookups
    num_tables: original table count T (preserved even after multi-hot expansion)
    row_shift:  bit offset separating table id and row id in keys
    """
    keys: np.ndarray
    valid: np.ndarray
    num_tables: int
    row_shift: int = 26

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @property
    def num_cols(self) -> int:
        return int(self.keys.shape[1])

    @property
    def raw_lookups(self) -> int:
        return int(self.valid.sum())


def pack_onehot(indices: np.ndarray, lengths: np.ndarray, row_shift: int = 26) -> LookupBatch:
    """Pack table-major one-hot (indices, lengths) into a sample-major LookupBatch."""
    if lengths.ndim != 2:
        raise ValueError(f"lengths must be 2D (T, B), got shape {lengths.shape}")
    T, B = lengths.shape
    if lengths.size and int(lengths.max()) > 1:
        raise ValueError("one-hot only: lengths.max() > 1")
    valid_tb = lengths > 0
    K_tb = np.full((T, B), -1, dtype=np.int64)
    if int(valid_tb.sum()) != len(indices):
        raise ValueError(
            f"indices length ({len(indices)}) does not match valid entries ({int(valid_tb.sum())})"
        )
    if len(indices):
        table_ids = np.repeat(np.arange(T, dtype=np.int64), valid_tb.sum(1))
        K_tb[valid_tb] = (table_ids << row_shift) | np.asarray(indices, dtype=np.int64)
    return LookupBatch(
        keys=K_tb.T.copy(),
        valid=valid_tb.T.copy(),
        num_tables=T,
        row_shift=row_shift,
    )


def load_onehot_batch(path: Path, row_shift: int = 26) -> LookupBatch:
    """Load a single one-hot batch file (b{i}.npz) into a LookupBatch."""
    with np.load(path) as d:
        return pack_onehot(d["indices"], d["lengths"], row_shift=row_shift)


def iter_batches(batch_dir: Path, steps: int, row_shift: int = 26) -> Iterator[tuple[int, LookupBatch]]:
    """Yield (step_idx, LookupBatch) for b0.npz .. b{steps-1}.npz."""
    batch_dir = Path(batch_dir)
    for i in range(steps):
        yield i, load_onehot_batch(batch_dir / f"b{i}.npz", row_shift=row_shift)


def scan_table_lengths(batch_dir: Path, steps: int) -> np.ndarray:
    """Scan b0.npz .. b{steps-1}.npz and return per-table (max_row_id + 1)."""
    batch_dir = Path(batch_dir)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    with np.load(batch_dir / "b0.npz") as d0:
        T = int(d0["lengths"].shape[0])
    tlen = np.zeros(T, dtype=np.int64)
    for i in range(steps):
        with np.load(batch_dir / f"b{i}.npz") as d:
            lengths = d["lengths"]
            indices = d["indices"]
            if len(indices):
                tables = np.repeat(np.arange(T, dtype=np.int64), (lengths > 0).sum(1))
                np.maximum.at(tlen, tables, np.asarray(indices, dtype=np.int64) + 1)
    return tlen


def compute_rowmax(keys: np.ndarray, num_tables: int, row_shift: int = 26) -> np.ndarray:
    """Return per-table maximum row id present in packed keys."""
    rowmax = np.zeros(num_tables, dtype=np.int64)
    if len(keys):
        row_mask = (1 << row_shift) - 1
        np.maximum.at(rowmax, keys >> row_shift, keys & row_mask)
    return rowmax


def save_step_cache(
    path: Path,
    *,
    keys: np.ndarray,
    counts: np.ndarray,
    mask: np.ndarray,
    gpu: np.ndarray,
    raw: int,
    num_tables: int,
    row_shift: int = 26,
) -> None:
    """Write a per-step placement cache readable by step_traffic.py and linkmodel.py."""
    rowmax = compute_rowmax(keys, num_tables=num_tables, row_shift=row_shift)
    np.savez(
        path,
        keys=keys,
        counts=counts,
        mask=mask,
        rowmax=rowmax,
        raw=np.int64(raw),
        gpu=gpu,
    )
