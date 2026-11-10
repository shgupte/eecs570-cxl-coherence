import pytest

from baseline import aggregate_gradients, apply_owner_updates, collect_unique_accesses, route_conventional, simulate
from config import SimConfig
from topology import Ownership, owner
from workload import synthetic_batch, training_steps


def _example_batch():
    # GPU 0: [10, 10, 20], GPU 1: [10, 30]. With 8 GPUs, row 10 is owned by GPU 2.
    return synthetic_batch({0: {0: [10, 10, 20], 1: [10, 30]}}, num_samples=8)


def _config(**overrides):
    values = dict(
        num_gpus=8,
        embedding_dim=4,
        bytes_per_element=4,
        request_bytes=8,
        track_versions=True,
        collect_rows=True,
    )
    values.update(overrides)
    return SimConfig(**values)


def test_two_gpus_send_separate_contributions_and_the_row_updates_once():
    batch = _example_batch()
    config = _config()
    results, versions = simulate(training_steps(batch, 8), config)
    step = results[0]
    assert step.remote_gradient_vectors == 4
    assert step.local_gradient_contributions == 0
    assert step.backward_vectors[0][2] == 1
    assert step.backward_vectors[1][2] == 1
    assert step.backward_bytes == 4 * 16
    assert step.updated_rows == 3
    assert step.updated_row_ids == {(0, 10), (0, 20), (0, 30)}
    assert step.contributors[(0, 10)] == (0, 1)
    assert owner(0, 10, 8) == 2
    assert versions[(0, 10)] == 1
    assert versions[(0, 20)] == 1
    assert (0, 11) not in versions
    assert (0, 11) not in step.updated_row_ids


def test_update_happens_after_aggregation_and_not_before():
    batch = _example_batch()
    step = training_steps(batch, 8)[0]
    accesses = collect_unique_accesses(batch, step.ranges)
    routed = route_conventional(accesses, Ownership.modulo(8))
    versions = {}
    aggregate_gradients(routed)
    assert versions == {}
    with pytest.raises(RuntimeError, match="after gradient"):
        apply_owner_updates(accesses, 8, versions, gradients_aggregated=False, collect_rows=True)
    assert versions == {}
    apply_owner_updates(accesses, 8, versions, gradients_aggregated=True, collect_rows=True)
    assert versions[(0, 10)] == 1
    assert len(versions) == 3


def test_same_row_in_two_tables_updates_twice_as_two_embeddings():
    batch = synthetic_batch({0: {0: [10]}, 1: {0: [10]}}, num_samples=8)
    _, versions = simulate(training_steps(batch, 8), _config())
    assert versions[(0, 10)] == 1
    assert versions[(1, 10)] == 1


def test_conventional_baseline_refetches_on_the_next_step():
    batch = _example_batch()
    results, versions = simulate(training_steps(batch, 8, repeats=2), _config())
    assert results[0].remote_requests == results[1].remote_requests == 4
    assert results[0].cache_hits == results[1].cache_hits == 0
    assert results[0].origin == "synthetic-repeat"
    assert versions[(0, 10)] == 2
    assert versions[(0, 20)] == 2
    assert results[1].updated_rows == 3
    # No persistent cache: forward bytes do not drop on the second step.
    assert results[1].forward_bytes == results[0].forward_bytes
    assert results[1].forward_bytes > 0
