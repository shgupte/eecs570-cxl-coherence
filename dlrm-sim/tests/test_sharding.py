from baseline import collect_unique_accesses, simulate
from config import SimConfig
from topology import owner
from workload import synthetic_batch, training_steps


def test_each_row_has_one_owner_and_table_id_does_not_change_it():
    assert owner(0, 10, 8) == 2
    assert owner(7, 10, 8) == owner(0, 10, 8)
    assert len({owner(table, 10, 8) for table in range(8)}) == 1
    for row in range(32):
        assert 0 <= owner(0, row, 8) < 8


def test_remote_access_communicates_and_local_access_does_not():
    # 8 samples, one per GPU. GPU 0 reads a local row and a remote row.
    bags = {0: {0: [16, 10], 2: [2]}}
    batch = synthetic_batch(bags, num_samples=8)
    config = SimConfig(num_gpus=8, embedding_dim=4, bytes_per_element=4, request_bytes=8)
    results, _ = simulate(training_steps(batch, 8), config)
    step = results[0]
    assert step.cache_hits == 0
    gpu0 = step.per_gpu[0]
    assert gpu0["local_requests"] == 1
    assert gpu0["remote_requests"] == 1
    assert step.forward_request_vectors[0][2] == 1
    assert step.forward_request_vectors[0][0] == 0
    assert step.forward_response_vectors[2][0] == 1
    assert step.forward_request_bytes == 8
    assert step.forward_response_bytes == 16
    # GPU 2's row 2 is local. No pair sends it anywhere.
    assert step.per_gpu[2]["local_requests"] == 1
    assert step.per_gpu[2]["remote_requests"] == 0
    assert all(value == 0 for value in step.forward_request_vectors[2])


def test_duplicates_inside_one_gpu_are_one_remote_request():
    bags = {0: {0: [10, 10, 20]}}
    batch = synthetic_batch(bags, num_samples=8)
    accesses = collect_unique_accesses(batch, training_steps(batch, 8)[0].ranges)
    assert accesses.per_gpu[0].raw_accesses == 3
    assert accesses.per_gpu[0].unique_rows[0].tolist() == [10, 20]
    config = SimConfig(num_gpus=8, embedding_dim=1, bytes_per_element=4, request_bytes=8)
    step = simulate(training_steps(batch, 8), config)[0][0]
    assert step.raw_accesses == 3
    assert step.unique_gpu_row_accesses == 2
    assert step.remote_requests == 2
    assert step.local_requests == 0
    assert step.forward_request_vectors[0][2] == 1
    assert step.forward_request_vectors[0][4] == 1
