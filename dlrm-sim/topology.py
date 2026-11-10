"""Row ownership for sharded embedding tables.

Each embedding is ``(table_id, row_id)``. The same row id in two tables is two
embeddings.

``contiguous`` is the TorchRec row-wise rule: a table's rows are split into
contiguous blocks, one per GPU, using the same sizes as
``RwSparseFeaturesDist`` (``ceil(num_rows / num_gpus)`` on the leading GPUs,
the remainder on the next GPU, and empty ranges when the table is smaller
than the GPU count).

``modulo`` is the earlier experimental rule, ``row_id % num_gpus``. It ignores
table size. Both models of one run must share one ``Ownership`` object.

The Meta access file does not store table cardinality. Contiguous sharding
uses ``max(observed row id) + 1`` unless a size is supplied. Rows past that
maximum are not in the trace, so the shard map covers the observed id span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from meta_trace import EmbeddingBatch


def even_shard_sizes(num_rows: int, num_gpus: int) -> list[int]:
    """Return one row-count per GPU, matching TorchRec's even row-wise split."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")
    if num_rows == 0:
        return [0] * num_gpus
    block = math.ceil(num_rows / num_gpus)
    occupied = num_rows // block
    sizes = [block] * occupied
    if num_rows % num_gpus != 0:
        remainder = num_rows - sum(sizes)
        if remainder > 0:
            sizes.append(remainder)
    if len(sizes) > num_gpus:
        raise RuntimeError("row split produced more shards than GPUs")
    sizes.extend([0] * (num_gpus - len(sizes)))
    return sizes


def owner(table_id: int, row_id: int, num_gpus: int) -> int:
    """Modulo owner. ``table_id`` is part of the embedding identity only."""
    del table_id
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    if row_id < 0:
        raise ValueError("row_id must be non-negative")
    return int(row_id) % num_gpus


def owners_of(row_ids: np.ndarray, num_gpus: int) -> np.ndarray:
    """Vector form of :func:`owner`."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    rows = np.asarray(row_ids)
    if rows.size and int(rows.min()) < 0:
        raise ValueError("row_id must be non-negative")
    return np.remainder(rows, num_gpus).astype(np.int64, copy=False)


def infer_table_rows(batch: EmbeddingBatch) -> dict[int, int]:
    """Table length implied by the trace: one past the largest row id seen."""
    sizes: dict[int, int] = {}
    for table_index, table_id in enumerate(batch.table_ids):
        window = batch.table_window(table_index, 0, batch.batch_size)
        sizes[int(table_id)] = int(window.max()) + 1 if window.size else 0
    return sizes


@dataclass
class Ownership:
    """One ownership map, shared by every communication model in a run."""

    scheme: str
    num_gpus: int
    table_rows: dict[int, int] = field(default_factory=dict)
    _edges: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.scheme not in {"contiguous", "modulo"}:
            raise ValueError("ownership scheme must be contiguous or modulo")
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be positive")
        if self.scheme == "contiguous":
            edges: dict[int, np.ndarray] = {}
            for table_id, num_rows in self.table_rows.items():
                sizes = even_shard_sizes(int(num_rows), self.num_gpus)
                edge = np.zeros(self.num_gpus + 1, dtype=np.int64)
                edge[1:] = np.cumsum(sizes, dtype=np.int64)
                edges[int(table_id)] = edge
            self._edges = edges

    @classmethod
    def modulo(cls, num_gpus: int) -> "Ownership":
        return cls(scheme="modulo", num_gpus=num_gpus)

    @classmethod
    def contiguous(cls, num_gpus: int, table_rows: dict[int, int]) -> "Ownership":
        return cls(scheme="contiguous", num_gpus=num_gpus, table_rows=dict(table_rows))

    def shard_sizes(self, table_id: int) -> list[int]:
        if self.scheme != "contiguous":
            raise ValueError("modulo ownership has no contiguous shard sizes")
        edges = self._edges[int(table_id)]
        return [int(edges[gpu + 1] - edges[gpu]) for gpu in range(self.num_gpus)]

    def owner_of(self, table_id: int, row_id: int) -> int:
        if row_id < 0:
            raise ValueError("row_id must be non-negative")
        if self.scheme == "modulo":
            return int(row_id) % self.num_gpus
        edges = self._edges.get(int(table_id))
        if edges is None:
            raise KeyError(f"no contiguous size for table {table_id}")
        if row_id >= int(edges[-1]):
            raise ValueError(f"row {row_id} is outside table {table_id} ({int(edges[-1])} rows)")
        return int(np.searchsorted(edges, row_id, side="right") - 1)

    def owners(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_ids)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.scheme == "modulo":
            return owners_of(rows, self.num_gpus)
        edges = self._edges.get(int(table_id))
        if edges is None:
            raise KeyError(f"no contiguous size for table {table_id}")
        if int(rows.min()) < 0 or int(rows.max()) >= int(edges[-1]):
            raise ValueError(f"row id is outside table {table_id}")
        return np.searchsorted(edges, rows, side="right") - 1
