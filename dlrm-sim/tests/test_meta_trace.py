import numpy as np
import pytest

from meta_trace import (
    TraceStructureError,
    batch_from_arrays,
    batch_from_payload,
    load_meta_batch,
    subset_batch,
)
from workload import lookups_for_sample, synthetic_batch


def test_synthetic_bags_keep_duplicates_and_table_identity():
    batch = synthetic_batch(
        {
            0: {0: [4, 4], 2: [1]},
            3: {0: [4], 1: []},
        },
        num_samples=3,
    )
    assert batch.table_ids == (0, 3)
    assert batch.batch_size == 3
    assert [lookup.row_id for lookup in batch.iter_lookups() if lookup.table_id == 0 and lookup.sample_id == 0] == [4, 4]
    assert lookups_for_sample(batch, 2) == [(0, 2, 1)]
    assert lookups_for_sample(batch, 1) == []
    assert batch.raw_accesses == 4
    # Same row id in two tables stays two embeddings.
    assert {(lookup.table_id, lookup.row_id) for lookup in batch.iter_lookups()} == {(0, 4), (0, 1), (3, 4)}


def test_lengths_must_be_two_dimensional():
    with pytest.raises(TraceStructureError, match="does not invent"):
        batch_from_arrays(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            source="flat",
            synthetic=True,
        )


def test_offset_mismatch_is_rejected():
    with pytest.raises(TraceStructureError, match="not recoverable"):
        batch_from_arrays(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([[1, 1]], dtype=np.int64),
            source="bad-offsets",
            synthetic=True,
        )


def test_unknown_payload_is_rejected():
    with pytest.raises(TraceStructureError, match="not invented"):
        batch_from_payload(("only-indices",), source="nope")


def test_float_indices_cast_when_integral():
    batch = batch_from_arrays(
        np.array([8.0, 8.0, 3.0], dtype=np.float32),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([[2, 1]], dtype=np.int64),
        source="float-indices",
        synthetic=False,
    )
    assert batch.indices.tolist() == [8, 8, 3]
    assert batch.index_storage == "float32"
    assert any("float32" in note or "integral" in note for note in batch.notes)


def test_non_integral_float_indices_are_rejected():
    with pytest.raises(TraceStructureError, match="non-integral"):
        batch_from_arrays(
            np.array([1.5], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([[1]], dtype=np.int64),
            source="fraction",
            synthetic=False,
        )


def test_subset_keeps_every_lookup_of_the_selected_samples():
    batch = synthetic_batch(
        {
            0: {0: [1, 1], 1: [2], 2: [3, 4]},
            5: {0: [9], 1: [8, 8], 2: [7]},
        }
    )
    subset = subset_batch(batch, table_ids=(5, 0), sample_start=1, sample_count=2)
    assert subset.table_ids == (5, 0)
    assert subset.sample_ids.tolist() == [1, 2]
    assert subset.raw_accesses == 1 + 2 + 2 + 1
    assert lookups_for_sample(subset, 1) == [(5, 1, 8), (5, 1, 8), (0, 1, 2)]
    assert lookups_for_sample(subset, 2) == [(5, 2, 7), (0, 2, 3), (0, 2, 4)]


def test_npz_roundtrip(tmp_path):
    batch = synthetic_batch({0: {0: [1, 2], 1: [2]}})
    path = tmp_path / "batch.npz"
    np.savez(
        path,
        indices=batch.indices,
        offsets=batch.offsets,
        lengths=batch.lengths,
    )
    loaded = load_meta_batch(path)
    assert loaded.raw_accesses == 3
    assert loaded.table_ids == (0,)
    assert list(loaded.iter_lookups())[0].row_id == 1
