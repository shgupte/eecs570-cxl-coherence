"""Load Meta embedding-bag access files into one training batch.

Observed file
--------------
``fbgemm_t856_bs65536_0.pt.gz`` from Meta's ``dlrm_datasets`` repository
(the only batch materialized on this machine; siblings are Git LFS pointers):

* payload is a 3-tuple ``(indices, offsets, lengths)``
* ``lengths`` is int32 with shape ``(num_tables, batch_size)`` = ``(856, 65536)``
* ``offsets`` is int64 with shape ``(num_tables * batch_size + 1,)``
* ``indices`` is float32 with shape ``(num_lookups,)``; stored values are
  whole numbers and are cast to int64

Bags are laid out table-major. Bag ``table_id * batch_size + sample_id``
occupies ``indices[offsets[bag] : offsets[bag + 1]]``, and that span's length
is ``lengths[table_id, sample_id]``. Duplicate row ids inside a bag are kept.
Empty bags have a zero length and an empty span.

``indices`` is float32 on disk. Integers above 2**24 are not all representable
in that dtype, so a published file can already have rounded large row ids.
This parser checks that every stored value is finite and integral, then casts
with truncation. It does not invent row ids the file cannot distinguish.

Other layouts are rejected. A 1-D ``lengths`` tensor does not say where tables
or samples start, and this loader will not guess those boundaries.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# float32 has a 23-bit mantissa; consecutive integers stop here.
FLOAT32_EXACT_INT_LIMIT = 2**24


class TraceStructureError(ValueError):
    """The file does not contain enough structure to rebuild samples."""


@dataclass(frozen=True)
class Lookup:
    """One embedding-bag index, in file order, duplicates included."""

    table_id: int
    sample_id: int
    row_id: int


@dataclass
class EmbeddingBatch:
    """One global training step's worth of embedding accesses.

    ``lengths[t, s]`` is the embedding bag for ``table_ids[t]`` and
    ``sample_ids[s]``. ``offsets`` is the CSR prefix of those bags in
    table-major order. ``indices`` holds the original row ids.
    """

    source: str
    synthetic: bool
    table_ids: tuple[int, ...]
    sample_ids: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    indices: np.ndarray
    index_storage: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        self.lengths = np.asarray(self.lengths)
        self.offsets = np.asarray(self.offsets)
        self.indices = np.asarray(self.indices)
        if self.offsets.dtype != np.int64:
            self.offsets = self.offsets.astype(np.int64)
        _validate_batch_arrays(self)

    @property
    def num_tables(self) -> int:
        return len(self.table_ids)

    @property
    def batch_size(self) -> int:
        return int(self.sample_ids.shape[0])

    @property
    def raw_accesses(self) -> int:
        return int(self.indices.shape[0])

    def bag_bounds(self, table_index: int, sample_index: int) -> tuple[int, int]:
        bag = table_index * self.batch_size + sample_index
        return int(self.offsets[bag]), int(self.offsets[bag + 1])

    def bag_rows(self, table_index: int, sample_index: int) -> np.ndarray:
        start, end = self.bag_bounds(table_index, sample_index)
        return _rows_as_int64(self.indices[start:end])

    def table_window(self, table_index: int, sample_start: int, sample_end: int) -> np.ndarray:
        """Row ids for a contiguous sample span of one table, duplicates kept.

        ``sample_end`` is exclusive and may equal ``batch_size``.
        """
        if not 0 <= sample_start <= sample_end <= self.batch_size:
            raise IndexError("sample span is outside the batch")
        start_bag = table_index * self.batch_size + sample_start
        end_bag = table_index * self.batch_size + sample_end
        start = int(self.offsets[start_bag])
        end = int(self.offsets[end_bag])
        return _rows_as_int64(self.indices[start:end])

    def iter_lookups(self):
        """Yield every lookup. Intended for tests and small batches."""
        for table_index, table_id in enumerate(self.table_ids):
            for sample_index, sample_id in enumerate(self.sample_ids.tolist()):
                rows = self.bag_rows(table_index, sample_index)
                for row_id in rows.tolist():
                    yield Lookup(int(table_id), int(sample_id), int(row_id))

    def describe(self) -> dict:
        return {
            "source": self.source,
            "synthetic": self.synthetic,
            "num_tables": self.num_tables,
            "batch_size": self.batch_size,
            "table_ids_head": list(self.table_ids[:8]),
            "sample_id_first": int(self.sample_ids[0]) if self.batch_size else None,
            "sample_id_last": int(self.sample_ids[-1]) if self.batch_size else None,
            "raw_accesses": self.raw_accesses,
            "index_storage": self.index_storage,
            "lengths_shape": list(self.lengths.shape),
            "offsets": int(self.offsets.shape[0]),
            "notes": list(self.notes),
        }


def _validate_batch_arrays(batch: EmbeddingBatch) -> None:
    if batch.batch_size < 1:
        raise TraceStructureError("batch has no samples")
    if batch.num_tables < 1:
        raise TraceStructureError("batch has no tables")
    if batch.lengths.shape != (batch.num_tables, batch.batch_size):
        raise TraceStructureError(
            f"lengths shape {batch.lengths.shape} does not match tables and samples"
        )
    if batch.lengths.dtype.kind not in "iu":
        raise TraceStructureError("lengths must be integers")
    if int(batch.lengths.min()) < 0:
        raise TraceStructureError("lengths must be non-negative")
    expected_offsets = batch.num_tables * batch.batch_size + 1
    if batch.offsets.shape != (expected_offsets,):
        raise TraceStructureError(
            f"offsets length {batch.offsets.shape[0]} != {expected_offsets}"
        )
    if int(batch.offsets[0]) != 0 or int(batch.offsets[-1]) != batch.indices.shape[0]:
        raise TraceStructureError("offsets do not span the indices array")
    flat_lengths = np.asarray(batch.lengths).reshape(-1)
    if not _spans_match(batch.offsets, flat_lengths):
        raise TraceStructureError(
            "offset spans do not match lengths; sample boundaries are not recoverable"
        )
    if batch.indices.ndim != 1:
        raise TraceStructureError("indices must be a flat array of row ids")
    if batch.indices.size and int(batch.indices.min()) < 0:
        raise TraceStructureError("row ids must be non-negative")
    if batch.sample_ids.shape != (batch.batch_size,):
        raise TraceStructureError("sample_ids must have one entry per sample")
    if np.any(np.diff(batch.sample_ids) != 1):
        raise TraceStructureError("sample ids must be a contiguous increasing range")


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(value)


def _describe_payload(payload) -> str:
    if isinstance(payload, dict):
        return "dict keys " + ", ".join(map(str, payload.keys()))
    if isinstance(payload, (tuple, list)):
        parts = []
        for item in payload:
            array = _as_numpy(item) if _is_array(item) else None
            if array is None:
                parts.append(type(item).__name__)
            else:
                parts.append(f"{type(item).__name__} shape {tuple(array.shape)} {array.dtype}")
        return f"{type(payload).__name__}[{', '.join(parts)}]"
    return type(payload).__name__


def _is_array(value) -> bool:
    return isinstance(value, np.ndarray) or hasattr(value, "shape") and hasattr(value, "dtype")


def _rows_as_int64(rows: np.ndarray) -> np.ndarray:
    """Cast one bag or GPU window. The full index array is not copied."""
    if rows.dtype == np.int64:
        return rows
    return rows.astype(np.int64, copy=False)


def _spans_match(offsets: np.ndarray, flat_lengths: np.ndarray) -> bool:
    count = int(flat_lengths.shape[0])
    if int(offsets.shape[0]) != count + 1:
        return False
    chunk = 1_000_000
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        spans = offsets[start + 1 : end + 1] - offsets[start:end]
        if not np.array_equal(spans, flat_lengths[start:end]):
            return False
    return True


def _integral_int64(indices: np.ndarray, storage: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Check row ids and keep the stored array.

    A full int64 copy of the Meta batch is about 3.7 GiB on top of the file's
    own tensors, and the login session kills the process near 4 GiB. Windows
    are cast when a GPU reads them.
    """
    notes: list[str] = []
    if indices.ndim != 1:
        raise TraceStructureError(f"indices must be rank-1, got shape {indices.shape}")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64), tuple(notes)

    if indices.dtype.kind == "f":
        notes.append(
            f"indices were stored as {storage}; values were checked to be finite "
            "and integral, and are cast to int64 per GPU window"
        )
        max_seen = 0.0
        chunk = 20_000_000
        for start in range(0, indices.size, chunk):
            part = indices[start : start + chunk]
            if not np.isfinite(part).all():
                raise TraceStructureError("indices contain a non-finite value")
            if not np.equal(part, np.trunc(part)).all():
                raise TraceStructureError("indices contain a non-integral value")
            if float(part.min()) < 0:
                raise TraceStructureError("row ids must be non-negative")
            max_seen = max(max_seen, float(part.max()))
        if max_seen >= FLOAT32_EXACT_INT_LIMIT and indices.dtype == np.float32:
            notes.append(
                "at least one row id is above 2**24, so the float32 file cannot "
                "represent every integer in that range; row ids are the stored values"
            )
        return indices, tuple(notes)
    if indices.dtype.kind in "iu":
        if int(indices.min()) < 0:
            raise TraceStructureError("row ids must be non-negative")
        return indices, tuple(notes)
    raise TraceStructureError(f"unsupported indices dtype {indices.dtype}")


def batch_from_arrays(
    indices,
    offsets,
    lengths,
    *,
    source: str,
    synthetic: bool,
    table_ids: tuple[int, ...] | None = None,
    sample_ids: np.ndarray | None = None,
    index_storage: str | None = None,
    notes: tuple[str, ...] = (),
) -> EmbeddingBatch:
    """Build a batch from CSR arrays. Lengths must be ``(tables, samples)``."""
    lengths_np = _as_numpy(lengths)
    offsets_np = _as_numpy(offsets)
    indices_np = _as_numpy(indices)
    if lengths_np.ndim != 2:
        raise TraceStructureError(
            "lengths must be 2-D (num_tables, batch_size) to recover sample and "
            f"table boundaries; got shape {tuple(lengths_np.shape)}. "
            "This loader does not invent those boundaries."
        )
    num_tables, batch_size = (int(lengths_np.shape[0]), int(lengths_np.shape[1]))
    if table_ids is None:
        table_ids = tuple(range(num_tables))
    if len(table_ids) != num_tables:
        raise TraceStructureError("table_ids length does not match lengths")
    if sample_ids is None:
        sample_ids = np.arange(batch_size, dtype=np.int64)
    storage = index_storage or str(indices_np.dtype)
    converted, cast_notes = _integral_int64(indices_np, storage)
    return EmbeddingBatch(
        source=source,
        synthetic=synthetic,
        table_ids=tuple(int(table_id) for table_id in table_ids),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        lengths=np.asarray(lengths_np, dtype=np.int64),
        offsets=np.asarray(offsets_np, dtype=np.int64),
        indices=converted,
        index_storage=storage,
        notes=tuple(notes) + cast_notes,
    )


def batch_from_payload(payload, *, source: str, synthetic: bool = False) -> EmbeddingBatch:
    """Accept the published 3-tensor payload, or a dict with the same fields."""
    if isinstance(payload, dict):
        try:
            indices = payload["indices"]
            offsets = payload["offsets"]
            lengths = payload["lengths"]
        except KeyError as exc:
            raise TraceStructureError(
                "dict payload is missing indices, offsets, or lengths "
                f"({_describe_payload(payload)})"
            ) from exc
        return batch_from_arrays(indices, offsets, lengths, source=source, synthetic=synthetic)

    if isinstance(payload, (tuple, list)) and len(payload) == 3 and all(_is_array(item) for item in payload):
        indices, offsets, lengths = payload
        return batch_from_arrays(indices, offsets, lengths, source=source, synthetic=synthetic)

    raise TraceStructureError(
        "unsupported embedding-access payload "
        f"({_describe_payload(payload)}). Expected (indices, offsets, lengths) "
        "with 2-D lengths of shape (num_tables, batch_size). "
        "Sample boundaries were not invented."
    )


def load_meta_batch(path: str | Path) -> EmbeddingBatch:
    """Load a ``.pt``, ``.pt.gz``, or ``.npz`` access file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    name = file_path.name
    if name.endswith(".npz"):
        with np.load(file_path) as data:
            missing = {"indices", "offsets", "lengths"} - set(data.files)
            if missing:
                raise TraceStructureError(
                    f"{file_path} is missing {sorted(missing)}; sample boundaries were not invented"
                )
            return batch_from_arrays(
                data["indices"],
                data["offsets"],
                data["lengths"],
                source=str(file_path),
                synthetic=False,
            )
    if name.endswith(".pt") or name.endswith(".pt.gz") or name.endswith(".gz"):
        import torch

        if name.endswith(".gz"):
            import gzip

            with gzip.open(file_path, "rb") as stream:
                payload = torch.load(stream, map_location="cpu")
        else:
            payload = torch.load(file_path, map_location="cpu")
        batch = batch_from_payload(payload, source=str(file_path), synthetic=False)
        del payload
        gc.collect()
        return batch
    raise TraceStructureError(f"unsupported file type: {file_path}")


def subset_batch(
    batch: EmbeddingBatch,
    *,
    table_ids: tuple[int, ...] | None = None,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> EmbeddingBatch:
    """Contiguous sample slice and an explicit table subset. Ids are preserved."""
    if sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    sample_end = batch.batch_size if sample_count is None else sample_start + sample_count
    if sample_count is not None and sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0 <= sample_start < sample_end <= batch.batch_size:
        raise ValueError("sample slice is outside the batch")

    if table_ids is None:
        chosen = tuple(range(batch.num_tables))
    else:
        position = {table_id: index for index, table_id in enumerate(batch.table_ids)}
        missing = [table_id for table_id in table_ids if table_id not in position]
        if missing:
            raise ValueError(f"unknown table ids {missing}")
        chosen = tuple(position[table_id] for table_id in table_ids)

    full = (
        chosen == tuple(range(batch.num_tables))
        and sample_start == 0
        and sample_end == batch.batch_size
    )
    if full:
        return batch

    new_lengths = []
    new_indices = []
    cursor = 0
    new_offsets = [0]
    width = sample_end - sample_start
    for table_index in chosen:
        window = batch.table_window(table_index, sample_start, sample_end)
        bag_lengths = batch.lengths[table_index, sample_start:sample_end]
        if int(bag_lengths.sum()) != int(window.shape[0]):
            raise TraceStructureError("subset window does not match bag lengths")
        new_lengths.append(np.asarray(bag_lengths, dtype=np.int64))
        new_indices.append(window)
        for length in bag_lengths.tolist():
            cursor += int(length)
            new_offsets.append(cursor)
    notes = batch.notes + (
        "subset of the source batch; this is not a claim that the slice is a separate measured step",
    )
    return EmbeddingBatch(
        source=batch.source,
        synthetic=batch.synthetic,
        table_ids=tuple(batch.table_ids[index] for index in chosen),
        sample_ids=batch.sample_ids[sample_start:sample_end].copy(),
        lengths=np.stack(new_lengths, axis=0) if new_lengths else np.zeros((0, width), dtype=np.int64),
        offsets=np.asarray(new_offsets, dtype=np.int64),
        indices=np.concatenate(new_indices) if new_indices else np.empty(0, dtype=np.int64),
        index_storage=batch.index_storage,
        notes=notes,
    )
