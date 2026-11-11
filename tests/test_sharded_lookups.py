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

    def test_remote_read_uses_owner_shard_and_owner_writes_once(self):
        # Two GPUs. With ids 0 and 5, the span is 6 rows: GPU 0 owns 0-2, GPU 1 owns 3-5.
        lookups = [
            (0, 0, 0),
            (0, 0, 5),
            (0, 0, 5),
            (1, 0, 0),
            (1, 0, 0),
        ]
        result = distribute_lookups(
            lookups,
            num_gpus=2,
            shard_bytes=4096,
            row_bytes=64,
            line_bytes=64,
            issue_gap=10,
            phase_gap=100,
        )
        self.assertEqual(result["local_row_reads"], [1, 0])
        self.assertEqual(result["remote_row_reads"], [2, 2])
        self.assertEqual(result["row_writes"], [1, 1])
        reads = [item for item in result["requests"][0] if item[0] == "r"]
        self.assertEqual(len(reads), 3)
        self.assertLess(reads[0][1], 4096)
        self.assertGreaterEqual(reads[1][1], 4096)
        self.assertEqual(reads[1][1], reads[2][1])
        owner_writes = [item for item in result["requests"][1] if item[0] == "w"]
        self.assertEqual(len(owner_writes), 1)
        self.assertGreaterEqual(owner_writes[0][1], 4096)
        self.assertGreaterEqual(owner_writes[0][3], result["write_start_tick"])
        self.assertGreater(result["write_start_tick"], result["read_done_tick"])


if __name__ == "__main__":
    unittest.main()
