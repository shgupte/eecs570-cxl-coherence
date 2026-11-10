import pytest

from coherent import simulate_coherent
from config import SimConfig
from topology import Ownership
from torchrec_baseline import simulate_torchrec
from workload import synthetic_batch, training_steps


def _batch():
    # 4 samples, one per GPU. Table length 1,000. Rows 300 and 400 belong to GPU 1.
    return synthetic_batch({0: {0: [10, 300, 300, 400], 1: [300]}}, num_samples=4)


def _ownership():
    return Ownership.contiguous(4, {0: 1000})


def _config(**overrides):
    values = dict(
        num_gpus=4,
        embedding_dim=4,
        bytes_per_element=4,
        request_bytes=8,
        track_versions=True,
        collect_rows=True,
        protocol="vi",
        invalidation_bytes=64,
        acknowledgement_bytes=64,
    )
    values.update(overrides)
    return SimConfig(**values)


def test_duplicate_in_one_step_hits_then_the_update_invalidates():
    batch = _batch()
    config = _config()
    steps = training_steps(batch, 4, repeats=2)
    results, versions, cache = simulate_coherent(steps, config, _ownership())
    first, second = results
    # GPU 0 sees remote rows 300, 300, 400. The second 300 hits.
    assert first.cache_misses == 2
    assert first.cache_hits == 1
    assert first.cache_evictions == 0
    assert first.invalidations == 2
    assert cache.resident == 0
    assert second.cache_hits == 1
    assert second.cache_misses == first.cache_misses
    assert versions[(0, 300)] == 2
    assert first.per_row_remote_gradient_vectors == 2
    assert first.forward_response_bytes == first.cache_misses * 16
    assert first.invalidation_bytes == 2 * 64
    assert first.acknowledgement_bytes == 2 * 64


def test_finite_cache_evicts_the_oldest_row():
    batch = _batch()
    config = _config(cache_bytes=16)
    step = simulate_coherent(training_steps(batch, 4), config, _ownership())[0][0]
    # Capacity is one row. 300 hits the second time, then 400 evicts it.
    assert config.cache_rows == 1
    assert step.cache_hits == 1
    assert step.cache_misses == 2
    assert step.cache_evictions == 1
    assert step.invalidations == 1
    assert step.per_row_remote_gradient_vectors == 2


def test_local_rows_do_not_enter_the_cache():
    batch = _batch()
    step = simulate_coherent(training_steps(batch, 4), _config(), _ownership())[0][0]
    # Row 10 is owned by GPU 0, which is the GPU for sample 0. Row 300 on GPU 1 is local.
    assert step.cache_misses + step.cache_hits == step.remote_requests
    assert step.forward_request_vectors[0][0] == 0


def test_updated_rows_match_the_sharded_run():
    batch = _batch()
    ownership = _ownership()
    config = _config()
    coherent = simulate_coherent(training_steps(batch, 4), config, ownership)[0][0]
    sharded = simulate_torchrec(training_steps(batch, 4), config, ownership)[0][0]
    assert coherent.updated_row_ids == sharded.updated_row_ids
    assert coherent.protocol == "vi"
    assert coherent.model == "coherent"


def test_unknown_protocol_and_tiny_cache_are_rejected():
    with pytest.raises(ValueError, match="protocol"):
        SimConfig(protocol="mesi")
    config = _config(cache_bytes=8)
    with pytest.raises(ValueError, match="smaller than one"):
        simulate_coherent(training_steps(_batch(), 4), config, _ownership())
