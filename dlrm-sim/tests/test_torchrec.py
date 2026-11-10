import numpy as np

from baseline import simulate
from config import SimConfig
from topology import Ownership, even_shard_sizes
from torchrec_baseline import simulate_torchrec
from workload import bags_for_gpu, synthetic_batch, training_steps


def _batch():
    # 4 samples, one per GPU. Table 0 has 1,000 rows. Table 1 has 10.
    return synthetic_batch(
        {
            0: {0: [10, 10, 300, 400], 1: [300, 300, 800]},
            1: {0: [9, 9], 2: [1]},
        },
        num_samples=4,
    )


def _ownership():
    return Ownership.contiguous(4, {0: 1000, 1: 10})


def _config():
    return SimConfig(
        num_gpus=4,
        embedding_dim=4,
        bytes_per_element=4,
        request_bytes=8,
        track_versions=True,
        collect_rows=True,
    )


def test_contiguous_shards_match_torchrec_block_sizes():
    assert even_shard_sizes(1000, 4) == [250, 250, 250, 250]
    assert even_shard_sizes(10, 4) == [3, 3, 3, 1]
    assert even_shard_sizes(5, 8) == [1, 1, 1, 1, 1, 0, 0, 0]
    ownership = _ownership()
    assert ownership.owner_of(0, 0) == 0
    assert ownership.owner_of(0, 249) == 0
    assert ownership.owner_of(0, 250) == 1
    assert ownership.owner_of(0, 749) == 2
    assert ownership.owner_of(0, 750) == 3
    assert ownership.owner_of(0, 999) == 3
    # Same row id, smaller table: row 9 is the last row of the size-1 tail.
    assert ownership.owner_of(1, 9) == 3
    assert ownership.owner_of(1, 0) == 0
    modulo = Ownership.modulo(4)
    assert modulo.owner_of(0, 10) == 2
    assert modulo.owner_of(0, 10) != ownership.owner_of(0, 10)


def test_bags_keep_duplicates_and_boundaries():
    batch = _batch()
    step = training_steps(batch, 4)[0]
    gpu0 = bags_for_gpu(batch, step.ranges[0])
    assert gpu0[0][0] == [10, 10, 300, 400]
    assert gpu0[0][1] == [9, 9]
    assert list(gpu0) == [0]


def test_one_bag_can_produce_one_partial_for_several_rows():
    batch = _batch()
    ownership = _ownership()
    config = _config()
    step = simulate_torchrec(training_steps(batch, 4), config, ownership)[0][0]
    # Sample 0 / table 0 rows 300 and 400 share GPU 1, so one remote partial.
    # The duplicate 10 stays on GPU 0 and does not add a second partial.
    assert step.forward_response_vectors[1][0] == 1
    # Two copies of 300 are two id requests, still one partial, and row 800 is another.
    assert step.forward_request_vectors[1][3] == 1
    assert step.id_request_count == (
        2  # sample 0: 300 and 400
        + 1  # sample 1: 800
        + 2  # sample 0 table 1: 9, 9
        + 1  # sample 2 table 1: 1, owned by GPU 0
    )
    assert step.remote_partial_pooled_vectors == 4
    assert step.partial_pooled_vectors == 6
    assert step.per_row_remote_gradient_vectors == 0
    assert step.remote_gradient_vectors == 0
    assert step.remote_pooled_gradient_vectors == step.remote_partial_pooled_vectors
    assert step.backward_vectors[1][3] == 1
    assert step.owner_lookups == 4 + 3 + 2 + 1
    assert step.row_gradient_applications == step.owner_lookups


def test_reduce_scatter_counts_empty_bags_and_is_not_the_partial_count():
    batch = _batch()
    config = _config()
    step = simulate_torchrec(training_steps(batch, 4), config, _ownership())[0][0]
    embedding_bytes = 16
    assert step.reduce_scatter_bytes == 3 * 4 * 2 * embedding_bytes
    assert step.allgather_backward_bytes == step.reduce_scatter_bytes
    assert step.forward_bytes == step.id_request_count * 8 + step.reduce_scatter_bytes
    assert step.backward_bytes == step.allgather_backward_bytes
    assert step.logical_remote_partial_bytes == step.remote_partial_pooled_vectors * embedding_bytes
    assert step.logical_remote_partial_bytes != step.reduce_scatter_bytes


def test_duplicate_and_shared_rows_update_once():
    batch = _batch()
    config = _config()
    ownership = _ownership()
    torchrec, versions = simulate_torchrec(training_steps(batch, 4), config, ownership)
    reference, reference_versions = simulate(training_steps(batch, 4), config, ownership)
    step = torchrec[0]
    assert step.updated_row_ids == reference[0].updated_row_ids
    assert step.updated_row_ids == {(0, 10), (0, 300), (0, 400), (0, 800), (1, 9), (1, 1)}
    assert step.contributors[(0, 300)] == (0, 1)
    assert versions[(0, 10)] == 1
    assert versions == reference_versions
    # Row-transfer still charges one gradient per unique remote row.
    assert reference[0].per_row_remote_gradient_vectors > 0
    assert reference[0].model == "row-transfer"
    assert step.model == "torchrec"
    assert reference[0].ownership_scheme == step.ownership_scheme == "contiguous"
    # Two rows on GPU 1 become two row-transfer responses and one pooled partial.
    assert reference[0].forward_response_vectors[1][0] == 2
    assert step.forward_response_vectors[1][0] == 1


def test_modulo_option_changes_owners_but_not_the_updated_set():
    batch = _batch()
    config = _config()
    contiguous = simulate_torchrec(training_steps(batch, 4), config, _ownership())[0][0]
    modulo_ownership = Ownership.modulo(4)
    modulo = simulate_torchrec(training_steps(batch, 4), config, modulo_ownership)[0][0]
    assert contiguous.updated_row_ids == modulo.updated_row_ids
    assert modulo.ownership_scheme == "modulo"
    assert modulo.id_request_count != contiguous.id_request_count
