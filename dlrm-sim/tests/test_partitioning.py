from workload import lookups_for_sample, partition_samples, synthetic_batch, training_steps


def _gpu_of(ranges, sample_id):
    owners = [item.gpu_id for item in ranges if item.sample_start <= sample_id < item.sample_end]
    assert len(owners) == 1
    return owners[0]


def test_each_sample_belongs_to_one_gpu_and_lookups_stay_together():
    bags = {
        table: {sample: [table * 100 + sample, sample] for sample in range(16)}
        for table in (0, 1, 4)
    }
    # A repeated index inside one sample must not be dropped or moved.
    bags[0][3] = [3, 3, 50]
    batch = synthetic_batch(bags, num_samples=16)
    ranges = partition_samples(batch, num_gpus=8)
    assert [(item.sample_start, item.sample_end) for item in ranges] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 10),
        (10, 12),
        (12, 14),
        (14, 16),
    ]
    seen = []
    for sample_id in range(16):
        lookups = lookups_for_sample(batch, sample_id)
        assert _gpu_of(ranges, sample_id) == sample_id // 2
        seen.extend(lookups)
    assert len(seen) == batch.raw_accesses
    assert seen.count((0, 3, 3)) == 2


def test_remainder_samples_stay_contiguous_and_cover_the_batch():
    batch = synthetic_batch({0: {sample: [sample] for sample in range(10)}})
    ranges = partition_samples(batch, num_gpus=4)
    assert [item.num_samples for item in ranges] == [3, 3, 2, 2]
    covered = []
    for item in ranges:
        covered.extend(range(item.sample_start, item.sample_end))
    assert covered == list(range(10))
    assert len(covered) == len(set(covered))


def test_partition_is_identical_across_steps():
    batch = synthetic_batch({0: {0: [1], 7: [2]}}, num_samples=8)
    steps = training_steps(batch, num_gpus=8, repeats=2)
    assert steps[0].ranges == steps[1].ranges
    assert steps[0].origin == "synthetic-repeat"
    assert steps[0].cross_step_locality == "synthetic-not-measured"
    assert all(item.num_samples == 1 for item in steps[0].ranges)
