#!/usr/bin/env python3
"""MLPerf DLRM-DCNv2 synthetic multi-hot expansion.

Every one-hot id maps to a deterministic fixed bag consisting of the id itself
(modulo table row count) plus size-1 extra rows drawn via a SplitMix64-style
hash of (table, id, slot), matching the uniform distribution of
facebookresearch/dlrm torchrec_dlrm/multi_hot.py.
"""
from dataclasses import dataclass
import numpy as np

from batch_io import LookupBatch

MH_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6,
    9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1,
]
MH_ROWS = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36,
]

_HASH_C1 = np.uint64(0x9E3779B97F4A7C15)
_HASH_C2 = np.uint64(0xBF58476D1CE4E5B9)
_HASH_C3 = np.uint64(0x94D049BB133111EB)
_HASH_MIX = np.uint64(0xD6E8FEB86659FD93)


@dataclass(frozen=True)
class MultiHotSpec:
    """Per-table bag sizes and row counts for deterministic multi-hot expansion."""
    sizes: tuple[int, ...]
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.rows):
            raise ValueError(
                f"sizes length ({len(self.sizes)}) != rows length ({len(self.rows)})"
            )
        if not self.sizes:
            raise ValueError("MultiHotSpec must define at least one table")
        if any(s < 1 for s in self.sizes):
            raise ValueError("all bag sizes must be >= 1")
        if any(r < 1 for r in self.rows):
            raise ValueError("all table row counts must be >= 1")

    @property
    def num_tables(self) -> int:
        return len(self.sizes)

    @property
    def total_lookups(self) -> int:
        return sum(self.sizes)

    @property
    def table_lengths(self) -> np.ndarray:
        return np.array(self.rows, dtype=np.int64)


MLPERF_DLRM_V2 = MultiHotSpec(sizes=tuple(MH_SIZES), rows=tuple(MH_ROWS))


def multihot(
    K: np.ndarray,
    valid: np.ndarray,
    row_shift: int,
    spec: MultiHotSpec = MLPERF_DLRM_V2,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand K (B, T) one-hot keys -> (B, sum(spec.sizes)) multi-hot keys and valid mask."""
    if K.ndim != 2 or valid.shape != K.shape:
        raise ValueError(f"K and valid must be matching 2D arrays, got {K.shape} and {valid.shape}")
    B, T = K.shape
    if T != spec.num_tables:
        raise ValueError(f"expected {spec.num_tables} tables for multi-hot spec, got {T}")
    max_rows = max(spec.rows)
    if max_rows > (1 << row_shift):
        raise ValueError(
            f"max table rows ({max_rows}) exceeds row_shift capacity ({1 << row_shift})"
        )

    row_mask = (1 << row_shift) - 1
    out_k = np.empty((B, spec.total_lookups), dtype=np.int64)
    out_v = np.empty((B, spec.total_lookups), dtype=bool)

    col = 0
    with np.errstate(over="ignore"):
        for t in range(T):
            m = spec.sizes[t]
            n = np.uint64(spec.rows[t])
            table_prefix = np.int64(t) << row_shift
            base = (K[:, t] & row_mask).astype(np.uint64) % n
            t_term = np.uint64(t) * _HASH_C3
            base_scaled = base * _HASH_C1 + t_term if m > 1 else None

            out_k[:, col] = table_prefix | base.astype(np.int64)
            for j in range(1, m):
                x = base_scaled + np.uint64(j) * _HASH_C2
                x ^= x >> np.uint64(31)
                x *= _HASH_MIX
                x ^= x >> np.uint64(32)
                out_k[:, col + j] = table_prefix | (x % n).astype(np.int64)

            out_v[:, col : col + m] = valid[:, t : t + 1]
            col += m

    return out_k, out_v


def expand_multihot(
    batch: LookupBatch,
    spec: MultiHotSpec = MLPERF_DLRM_V2,
) -> LookupBatch:
    """Return a new LookupBatch with each one-hot feature expanded to a multi-hot bag."""
    K, valid = multihot(batch.keys, batch.valid, batch.row_shift, spec=spec)
    return LookupBatch(
        keys=K,
        valid=valid,
        num_tables=batch.num_tables,
        row_shift=batch.row_shift,
    )
