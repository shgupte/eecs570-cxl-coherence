import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_sharded_lookups import (
    distribute_lookups,
    even_shard_sizes,
    owner_of,
    owner_ranges,
    partition_samples,
)


class ShardedLookupTests(unittest.TestCase):
    def test_samples_and_rows_split_contiguously(self):
        self.assertEqual(partition_samples(0, 8, 4), [(0, 2), (2, 4), (4, 6), (6, 8)])
        self.assertEqual(even_shard_sizes(1000, 4), [250, 250, 250, 250])
        ranges = owner_ranges(10, 4)
        self.assertEqual([owner_of(row, ranges) for row in range(10)], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3])

    def test_remote_bag_sends_ids_one_result_and_one_gradient(self):
        # GPU 0 owns rows 0-2, GPU 1 owns rows 3-5. One bag on GPU 0 contains
        # a local row and two copies of a remote row. One bag on GPU 1 contains
        # two copies of GPU 0's row.
        lookups = [
            (0, 0, 0, 0),
            (0, 0, 0, 5),
            (0, 0, 0, 5),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
        ]
        result = distribute_lookups(
            lookups,
            num_gpus=2,
            shard_bytes=4096,
            row_bytes=64,
            line_bytes=64,
            id_bytes=8,
            issue_gap=10,
            phase_gap=100,
        )
        self.assertEqual(result["id_sends"], [2, 2])
        self.assertEqual(result["pooled_results"], [1, 1])
        self.assertEqual(result["gradients"], [1, 1])
        self.assertEqual(result["row_writes"], [1, 1])
        self.assertEqual(result["local_row_reads"], [3, 2])
        gpu0 = result["requests"][0]
        remote_reads = [item for item in gpu0 if item[0] == "r" and item[1] >= 4096]
        self.assertEqual(remote_reads, [])
        id_writes = [item for item in gpu0 if item[0] == "w" and item[1] >= 4096 and item[2] == 8]
        self.assertEqual(len(id_writes), 2)
        gradients = [
            item for item in gpu0
            if item[0] == "w" and item[2] == 64 and item[1] >= 4096
            and item[3] >= result["gradient_start_tick"]
        ]
        self.assertEqual(len(gradients), 1)
        self.assertGreater(result["gradient_start_tick"], result["phase_starts"][2])
        self.assertGreater(result["update_start_tick"], result["gradient_start_tick"])


if __name__ == "__main__":
    unittest.main()
