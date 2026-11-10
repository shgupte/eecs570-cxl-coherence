"""Training steps and the GPU sample partition.

One ``EmbeddingBatch`` is one global training step. Samples are contiguous and
disjoint. Every lookup of a sample stays on the GPU that owns that sample.

Repeating one batch is a synthetic sequence. It is not a measurement of
production temporal locality. The downloaded Meta directory has a single real
batch (``fbgemm_t856_bs65536_0.pt.gz``); the other ``.pt.gz`` names are Git LFS
pointers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meta_trace import EmbeddingBatch, batch_from_arrays


@dataclass(frozen=True)
class SampleRange:
    """Half-open sample-id interval assigned to one GPU."""

    gpu_id: int
    sample_start: int
    sample_end: int

    @property
    def num_samples(self) -> int:
        return self.sample_end - self.sample_start


@dataclass(frozen=True)
class TrainingStep:
    """The sample partition shared by every access model."""

    step_index: int
    batch: EmbeddingBatch
    ranges: tuple[SampleRange, ...]
    origin: str
    cross_step_locality: str


def partition_samples(batch: EmbeddingBatch, num_gpus: int) -> tuple[SampleRange, ...]:
    """Split samples into contiguous GPU ranges.

    When ``batch_size`` is not divisible by ``num_gpus``, the first
    ``batch_size % num_gpus`` GPUs receive one extra sample. Ranges stay
    contiguous, disjoint, and complete. An empty range is possible only when
    there are more GPUs than samples.
    """
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    batch_size = batch.batch_size
    base, remainder = divmod(batch_size, num_gpus)
    first = int(batch.sample_ids[0])
    ranges = []
    cursor = 0
    for gpu_id in range(num_gpus):
        count = base + (1 if gpu_id < remainder else 0)
        ranges.append(
            SampleRange(
                gpu_id=gpu_id,
                sample_start=first + cursor,
                sample_end=first + cursor + count,
            )
        )
        cursor += count
    if cursor != batch_size:
        raise RuntimeError("sample partition did not cover the batch")
    return tuple(ranges)


def column_span(batch: EmbeddingBatch, sample_range: SampleRange) -> tuple[int, int]:
    """Convert an original-id range into column indexes of ``batch``."""
    first = int(batch.sample_ids[0])
    start = sample_range.sample_start - first
    end = sample_range.sample_end - first
    if start < 0 or end > batch.batch_size or start > end:
        raise ValueError("sample range is outside the batch")
    return start, end


def training_steps(
    batch: EmbeddingBatch,
    num_gpus: int,
    *,
    repeats: int = 1,
    origin: str | None = None,
) -> tuple[TrainingStep, ...]:
    """Build one or more steps that share a single sample partition.

    ``repeats > 1`` replays the same accesses. The origin is forced to
    ``synthetic-repeat`` and cross-step locality is marked unmeasured.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    ranges = partition_samples(batch, num_gpus)
    if repeats == 1:
        step_origin = origin or ("synthetic" if batch.synthetic else "meta-batch")
        locality = "not-applicable"
    else:
        step_origin = "synthetic-repeat"
        locality = "synthetic-not-measured"
    return tuple(
        TrainingStep(
            step_index=index,
            batch=batch,
            ranges=ranges,
            origin=step_origin,
            cross_step_locality=locality,
        )
        for index in range(repeats)
    )


def synthetic_batch(
    bags: dict[int, dict[int, list[int]]],
    *,
    num_samples: int | None = None,
    source: str = "synthetic",
) -> EmbeddingBatch:
    """Build a batch from ``bags[table_id][sample_id] = [row_id, ...]``.

    Missing bags are empty. Sample ids must be in ``0 .. num_samples-1``.
    Duplicate row ids are preserved, in the order given.
    """
    if not bags:
        raise ValueError("synthetic batch needs at least one table")
    observed = [sample_id for per_table in bags.values() for sample_id in per_table]
    if any(sample_id < 0 for sample_id in observed):
        raise ValueError("sample ids must be non-negative")
    if num_samples is None:
        num_samples = (max(observed) + 1) if observed else 1
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if any(sample_id >= num_samples for sample_id in observed):
        raise ValueError("sample id is outside num_samples")

    table_ids = tuple(sorted(int(table_id) for table_id in bags))
    lengths = np.zeros((len(table_ids), num_samples), dtype=np.int64)
    pieces: list[np.ndarray] = []
    offsets = [0]
    for table_index, table_id in enumerate(table_ids):
        per_sample = bags[table_id]
        for sample_id in range(num_samples):
            rows = per_sample.get(sample_id, [])
            if any(int(row_id) < 0 for row_id in rows):
                raise ValueError("row ids must be non-negative")
            array = np.asarray(rows, dtype=np.int64)
            lengths[table_index, sample_id] = array.size
            if array.size:
                pieces.append(array)
            offsets.append(offsets[-1] + int(array.size))
    indices = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    return batch_from_arrays(
        indices,
        np.asarray(offsets, dtype=np.int64),
        lengths,
        source=source,
        synthetic=True,
        table_ids=table_ids,
        index_storage="int64",
    )


def bags_for_gpu(
    batch: EmbeddingBatch,
    sample_range: SampleRange,
) -> dict[int, dict[int, list[int]]]:
    """``sample_id -> table_id -> row ids`` for one GPU.

    Bag order and duplicate row ids are kept. This is the logical access
    structure. The full Meta batch is not materialized this way; the simulator
    reads the same bags from the CSR arrays.
    """
    first = int(batch.sample_ids[0])
    start = sample_range.sample_start - first
    end = sample_range.sample_end - first
    if start < 0 or end > batch.batch_size:
        raise ValueError("sample range is outside the batch")
    nested: dict[int, dict[int, list[int]]] = {}
    for column in range(start, end):
        sample_id = int(batch.sample_ids[column])
        per_table: dict[int, list[int]] = {}
        for table_index, table_id in enumerate(batch.table_ids):
            per_table[int(table_id)] = [int(row) for row in batch.bag_rows(table_index, column).tolist()]
        nested[sample_id] = per_table
    return nested


def lookups_for_sample(batch: EmbeddingBatch, sample_id: int) -> list[tuple[int, int, int]]:
    """Every ``(table_id, sample_id, row_id)`` for one sample, bag order kept."""
    columns = np.flatnonzero(batch.sample_ids == sample_id)
    if columns.size != 1:
        raise KeyError(f"sample {sample_id} is not in the batch")
    column = int(columns[0])
    found = []
    for table_index, table_id in enumerate(batch.table_ids):
        for row_id in batch.bag_rows(table_index, column).tolist():
            found.append((int(table_id), int(sample_id), int(row_id)))
    return found
