#!/usr/bin/env python3
"""Unit, differential, C-backend parity, and downstream integration tests
for the refactored sample placement modules (batch_io, multihot, placement,
sample_placement).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import batch_io
import multihot as mh
import placement as pl
import sample_placement as sp


def _make_synthetic_batches(
    batch_dir: Path,
    *,
    steps: int = 5,
    B: int = 128,
    T: int = 26,
    seed: int = 42,
    missing_prob: float = 0.08,
) -> None:
    """Create deterministic table-major one-hot b{i}.npz files matching criteo_to_batches.py."""
    rng = np.random.default_rng(seed)
    batch_dir.mkdir(parents=True, exist_ok=True)
    # Use small-to-medium skewed row pools so some rows are shared (count 2..cmax),
    # some are hot (> cmax), and some are unique (count == 1).
    for i in range(steps):
        valid = rng.random((T, B)) >= missing_prob
        rows = np.zeros((T, B), dtype=np.int64)
        for t in range(T):
            cap = min(mh.MH_ROWS[t], 200 if t % 3 == 0 else 4000)
            # Zipf-like skew: mix a small hot set with wider IDs
            raw = rng.zipf(1.4, size=B) - 1
            rows[t] = raw % cap
        lengths = valid.astype(np.int32)
        indices = rows[valid].astype(np.int64)
        offsets = np.r_[0, np.cumsum(lengths.ravel())].astype(np.int64)
        np.savez(batch_dir / f"b{i}.npz", indices=indices, offsets=offsets, lengths=lengths)


def _compile_placer_for_test(tmpdir: Path) -> Path | None:
    """Compile analysis/placer.c into a loadable shared library using gcc or MSVC."""
    src = ANALYSIS_DIR / "placer.c"
    if not src.exists():
        return None
    if shutil.which("gcc"):
        so = tmpdir / ("placer.dll" if os.name == "nt" else "placer.so")
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
        return so
    if os.name == "nt":
        # Search standard Visual Studio BuildTools installations for vcvars64.bat
        vs_roots = [
            Path(r"C:\Program Files\Microsoft Visual Studio"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        ]
        vcvars = None
        for root in vs_roots:
            if root.exists():
                matches = sorted(root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"), reverse=True)
                if matches:
                    vcvars = matches[0]
                    break
        if vcvars is not None:
            dll = tmpdir / "placer.dll"
            bat = tmpdir / "build_placer.bat"
            bat.write_text(
                f'@echo off\r\n'
                f'call "{vcvars}" >nul\r\n'
                f'cl /nologo /O2 /LD "{src}" /Fe:"{dll}" /Fo:"{tmpdir / "placer.obj"}" '
                f'/link /EXPORT:assign >nul\r\n',
                encoding="utf-8",
            )
            res = subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True)
            if res.returncode == 0 and dll.exists():
                return dll
    return None


def _extract_original_script(dest: Path) -> bool:
    """Extract HEAD:analysis/sample_placement.py from git for golden differential testing."""
    res = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "HEAD:analysis/sample_placement.py"],
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout:
        return False
    dest.write_bytes(res.stdout)
    return True


class TestBatchIO(unittest.TestCase):
    def test_pack_and_load_onehot_with_missing_entries(self):
        # T=3 tables, B=4 samples
        lengths = np.array(
            [
                [1, 0, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 1],
            ],
            dtype=np.int32,
        )
        indices = np.array([10, 11, 12, 5, 6, 7], dtype=np.int64)
        shift = 26

        batch = batch_io.pack_onehot(indices, lengths, row_shift=shift)
        self.assertEqual(batch.batch_size, 4)
        self.assertEqual(batch.num_cols, 3)
        self.assertEqual(batch.num_tables, 3)
        self.assertEqual(batch.raw_lookups, 6)
        np.testing.assert_array_equal(batch.valid, lengths.T > 0)

        expected_k = np.array(
            [
                [(0 << shift) | 10, -1, (2 << shift) | 5],
                [-1, -1, (2 << shift) | 6],
                [(0 << shift) | 11, -1, -1],
                [(0 << shift) | 12, -1, (2 << shift) | 7],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(batch.keys, expected_k)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "b0.npz"
            np.savez(p, indices=indices, lengths=lengths)
            loaded = batch_io.load_onehot_batch(p, row_shift=shift)
            np.testing.assert_array_equal(loaded.keys, batch.keys)
            np.testing.assert_array_equal(loaded.valid, batch.valid)

    def test_pack_onehot_rejects_multihot_lengths(self):
        lengths = np.array([[2, 1]], dtype=np.int32)
        indices = np.array([1, 2, 3], dtype=np.int64)
        with self.assertRaises(ValueError):
            batch_io.pack_onehot(indices, lengths)

    def test_scan_table_lengths_handles_empty_tables(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # Step 0: table 1 empty
            np.savez(
                d / "b0.npz",
                lengths=np.array([[1, 1], [0, 0], [1, 0]], dtype=np.int32),
                indices=np.array([4, 9, 12], dtype=np.int64),
            )
            # Step 1: table 1 has row 7, table 0 has smaller row
            np.savez(
                d / "b1.npz",
                lengths=np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int32),
                indices=np.array([2, 7], dtype=np.int64),
            )
            tlen = batch_io.scan_table_lengths(d, steps=2)
            np.testing.assert_array_equal(tlen, np.array([10, 8, 13], dtype=np.int64))

    def test_save_step_cache_schema_and_rowmax_tables(self):
        shift = 26
        keys = np.array([(0 << shift) | 5, (0 << shift) | 42, (25 << shift) | 11], dtype=np.int64)
        counts = np.array([2, 3, 1], dtype=np.int64)
        mask = np.array([0b01, 0b11, 0b10], dtype=np.uint8)
        gpu = np.array([0, 1, 0, 1], dtype=np.int8)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "b0.npz"
            batch_io.save_step_cache(
                out, keys=keys, counts=counts, mask=mask, gpu=gpu,
                raw=6, num_tables=26, row_shift=shift,
            )
            with np.load(out) as z:
                self.assertEqual(set(z.files), {"keys", "counts", "mask", "rowmax", "raw", "gpu"})
                self.assertEqual(z["rowmax"].shape, (26,))
                self.assertEqual(z["rowmax"][0], 42)
                self.assertEqual(z["rowmax"][25], 11)
                self.assertEqual(int(z["raw"]), 6)


class TestMultiHot(unittest.TestCase):
    @staticmethod
    def _reference_multihot(K: np.ndarray, valid: np.ndarray, row_shift: int):
        """Exact copy of the original sample_placement.py multihot() loop."""
        B, T = K.shape
        cols, vals = [], []
        with np.errstate(over="ignore"):
            for t in range(T):
                n = np.uint64(mh.MH_ROWS[t])
                base = (K[:, t] & ((1 << row_shift) - 1)).astype(np.uint64) % n
                for j in range(mh.MH_SIZES[t]):
                    if j == 0:
                        r = base
                    else:
                        x = (
                            base * np.uint64(0x9E3779B97F4A7C15)
                            + np.uint64(j) * np.uint64(0xBF58476D1CE4E5B9)
                            + np.uint64(t) * np.uint64(0x94D049BB133111EB)
                        )
                        x ^= x >> np.uint64(31)
                        x *= np.uint64(0xD6E8FEB86659FD93)
                        x ^= x >> np.uint64(32)
                        r = x % n
                    cols.append((np.int64(t) << row_shift) | r.astype(np.int64))
                    vals.append(valid[:, t])
        return np.stack(cols, 1), np.stack(vals, 1)

    def test_vectorized_multihot_matches_original_bit_for_bit(self):
        rng = np.random.default_rng(123)
        B, T, shift = 64, 26, 26
        rows = rng.integers(0, 50_000_000, size=(B, T), dtype=np.int64)
        # Include boundary values: 0, MH_ROWS[t]-1, MH_ROWS[t]
        rows[0, :] = 0
        rows[1, :] = np.array(mh.MH_ROWS, dtype=np.int64) - 1
        rows[2, :] = np.array(mh.MH_ROWS, dtype=np.int64)
        valid = rng.random((B, T)) > 0.1
        K = np.where(valid, (np.arange(T, dtype=np.int64)[None, :] << shift) | rows, -1)

        ref_k, ref_v = self._reference_multihot(K, valid, shift)
        got_k, got_v = mh.multihot(K, valid, shift)

        self.assertEqual(got_k.shape, (B, 214))
        np.testing.assert_array_equal(got_k, ref_k)
        np.testing.assert_array_equal(got_v, ref_v)

    def test_expand_multihot_preserves_num_tables_and_valid_propagation(self):
        shift = 26
        B, T = 8, 26
        valid = np.ones((B, T), dtype=bool)
        valid[3, 20] = False  # Table 20 has bag size 100
        K = (np.arange(T, dtype=np.int64)[None, :] << shift) | np.arange(B, dtype=np.int64)[:, None]
        K[~valid] = -1
        batch = batch_io.LookupBatch(keys=K, valid=valid, num_tables=T, row_shift=shift)

        expanded = mh.expand_multihot(batch)
        self.assertEqual(expanded.num_tables, 26)
        self.assertEqual(expanded.num_cols, 214)
        self.assertEqual(expanded.batch_size, B)
        # Check table 20's 100 columns are all False for sample 3 and True for sample 0
        col_start = sum(mh.MH_SIZES[:20])
        col_end = col_start + mh.MH_SIZES[20]
        self.assertFalse(expanded.valid[3, col_start:col_end].any())
        self.assertTrue(expanded.valid[0, col_start:col_end].all())
        # Check all valid expanded rows stay within [0, MH_ROWS[t])
        col = 0
        row_mask = (1 << shift) - 1
        for t, (m, rmax) in enumerate(zip(mh.MH_SIZES, mh.MH_ROWS)):
            sub_k = expanded.keys[:, col : col + m]
            self.assertTrue(np.all((sub_k >> shift) == t))
            self.assertTrue(np.all((sub_k & row_mask) < rmax))
            col += m

    def test_multihot_validation_errors(self):
        K = np.zeros((4, 10), dtype=np.int64)
        V = np.ones((4, 10), dtype=bool)
        with self.assertRaises(ValueError):
            mh.multihot(K, V, row_shift=26)
        K26 = np.zeros((4, 26), dtype=np.int64)
        V26 = np.ones((4, 26), dtype=bool)
        with self.assertRaises(ValueError):
            # 1 << 20 = 1,048,576 < 40,000,000
            mh.multihot(K26, V26, row_shift=20)


class TestPlacementLogic(unittest.TestCase):
    def test_sole_updater_tracker_semantics_and_lag(self):
        tr = pl.SoleUpdaterTracker(lag=2)
        keys0 = np.array([10, 20, 30], dtype=np.int64)
        # Step 0: key 10 solely GPU 0, key 20 shared by GPU 0&1, key 30 solely GPU 2
        mask0 = np.array([0b001, 0b011, 0b100], dtype=np.uint8)
        self.assertIsNone(tr.bonus_for(keys0))
        tr.update(keys0, mask0)
        # Still fewer than lag=2 steps completed
        self.assertIsNone(tr.bonus_for(keys0))

        # Step 1: key 10 now shared (clears sole), key 20 untouched, key 40 solely GPU 1
        keys1 = np.array([10, 40], dtype=np.int64)
        mask1 = np.array([0b011, 0b010], dtype=np.uint8)
        tr.update(keys1, mask1)

        # At step 2 with lag=2, bonus must reflect step 0 state (not step 1)!
        probe = np.array([10, 20, 30, 40, 99], dtype=np.int64)
        bonus2 = tr.bonus_for(probe)
        np.testing.assert_array_equal(bonus2, np.array([0b001, 0, 0b100, 0, 0], dtype=np.uint8))

        # Step 2 update: key 30 untouched, key 40 untouched
        tr.update(np.array([99], dtype=np.int64), np.array([0b001], dtype=np.uint8))
        # At step 3 with lag=2, bonus reflects step 1 state:
        # key 10 cleared -> 0, key 20 still 0, key 30 retained 0b100, key 40 -> 0b010, key 99 not yet seen -> 0
        bonus3 = tr.bonus_for(probe)
        np.testing.assert_array_equal(bonus3, np.array([0, 0, 0b100, 0b010, 0], dtype=np.uint8))

    def test_greedy_colocates_shared_and_ignores_hot_rows(self):
        # 4 samples, 2 GPUs (cap=2).
        # Samples 0 & 2 share rows 100,101 (count=2 <= cmax=2).
        # Samples 1 & 3 share rows 200,201 (count=2 <= cmax=2).
        # All 4 share row 999 (count=4 > cmax=2, should get weight 0).
        K = np.array(
            [
                [100, 101, 999],
                [200, 201, 999],
                [100, 101, 999],
                [200, 201, 999],
            ],
            dtype=np.int64,
        )
        V = np.ones_like(K, dtype=bool)
        batch = batch_io.LookupBatch(keys=K, valid=V, num_tables=1, row_shift=26)

        placer = pl.SamplePlacer(pl.PlacementConfig(policy="greedy", gpus=2, cmax=2, lam=1.0))
        res = placer.place(batch)
        self.assertEqual(res.gpu[0], res.gpu[2])
        self.assertEqual(res.gpu[1], res.gpu[3])
        self.assertNotEqual(res.gpu[0], res.gpu[1])
        np.testing.assert_array_equal(np.bincount(res.gpu, minlength=2), [2, 2])

    def test_refine_passes_and_prior_steering(self):
        # 1. Verify prior mask directly steers assign()
        rid = np.array([[0], [1], [0], [1]], dtype=np.int64)
        valid = np.ones((4, 1), dtype=bool)
        w = np.ones(2, dtype=np.float32)
        g_no_prior, _ = pl.assign(rid, valid, 2, 2, w, 1.0)
        g_with_prior, _ = pl.assign(rid, valid, 2, 2, w, 1.0, prior=np.array([0b10, 0b01], dtype=np.uint8))
        np.testing.assert_array_equal(g_no_prior, np.array([0, 1, 0, 1], dtype=np.int8))
        np.testing.assert_array_equal(g_with_prior, np.array([1, 0, 1, 0], dtype=np.int8))

        # 2. Verify SamplePlacer runs 1 + passes backend calls with prior=cur for refine,
        #    and exactly 1 call with prior=None for greedy.
        class SpyBackend(pl.PythonBackend):
            def __init__(self):
                self.calls = []

            def assign(self, *args, **kwargs):
                self.calls.append(kwargs.get("prior"))
                return super().assign(*args, **kwargs)

        batch = batch_io.LookupBatch(keys=rid, valid=valid, num_tables=1, row_shift=26)

        spy_g = SpyBackend()
        pl.SamplePlacer(pl.PlacementConfig(policy="greedy", gpus=2, passes=3), backend=spy_g).place(batch)
        self.assertEqual(len(spy_g.calls), 1)
        self.assertIsNone(spy_g.calls[0])

        spy_r = SpyBackend()
        pl.SamplePlacer(pl.PlacementConfig(policy="refine", gpus=2, passes=2), backend=spy_r).place(batch)
        self.assertEqual(len(spy_r.calls), 3)
        self.assertIsNone(spy_r.calls[0])
        self.assertIsNotNone(spy_r.calls[1])
        self.assertIsNotNone(spy_r.calls[2])

    def test_affinity_and_owner_steering(self):
        shift = 26
        # 4 samples, 2 GPUs, 1 table of length 20 (owner 0 = rows 0..9, owner 1 = rows 10..19)
        tlen = np.array([20], dtype=np.int64)
        K = np.array([[15], [2], [16], [3]], dtype=np.int64)
        V = np.ones_like(K, dtype=bool)
        batch = batch_io.LookupBatch(keys=K, valid=V, num_tables=1, row_shift=shift)

        # High owner weight steers rows 15,16 to GPU 1 and rows 2,3 to GPU 0
        placer_own = pl.SamplePlacer(
            pl.PlacementConfig(policy="greedy", gpus=2, owner_weight=10.0, lam=0.5),
            tlen=tlen,
        )
        res_own = placer_own.place(batch)
        np.testing.assert_array_equal(res_own.gpu, np.array([1, 0, 1, 0], dtype=np.int8))

        # High affinity beta steers step 1 samples to whichever GPU solely updated their row in step 0
        placer_aff = pl.SamplePlacer(
            pl.PlacementConfig(policy="affinity", gpus=2, passes=0, beta=10.0, lam=0.5, lag=1)
        )
        res0 = placer_aff.place(batch)
        # Reorder samples in step 1: reverse order, each sample's unique row should follow its step-0 GPU
        batch_rev = batch_io.LookupBatch(keys=K[::-1], valid=V[::-1], num_tables=1, row_shift=shift)
        res1 = placer_aff.place(batch_rev)
        np.testing.assert_array_equal(res1.gpu, res0.gpu[::-1])

    def test_guardrails_and_edge_cases(self):
        with self.assertRaises(ValueError):
            pl.PlacementConfig(policy="greedy", gpus=9)
        with self.assertRaises(ValueError):
            pl.PlacementConfig(policy="greedy", gpus=0)
        with self.assertRaises(ValueError):
            pl.PlacementConfig(policy="affinity", gpus=4, lag=0)

        # B not divisible by G
        K = np.zeros((5, 2), dtype=np.int64)
        V = np.ones((5, 2), dtype=bool)
        batch = batch_io.LookupBatch(keys=K, valid=V, num_tables=2)
        placer = pl.SamplePlacer(pl.PlacementConfig(policy="greedy", gpus=2))
        with self.assertRaises(ValueError):
            placer.place(batch)

        # All-invalid batch should not crash
        K_empty = np.full((4, 2), -1, dtype=np.int64)
        V_empty = np.zeros((4, 2), dtype=bool)
        res = placer.place(batch_io.LookupBatch(keys=K_empty, valid=V_empty, num_tables=2))
        self.assertEqual(res.unique_rows, 0)
        self.assertEqual(res.raw, 0)
        self.assertEqual(len(res.gpu), 4)

    def test_owner_with_multihot_uses_mh_rows_safely(self):
        # One-hot batch has tiny row IDs (0..5), but multi-hot hashes up to MH_ROWS[t]-1 (up to 40M).
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            bdir, odir = d / "batches", d / "out"
            bdir.mkdir()
            T, B = 26, 16
            lengths = np.ones((T, B), dtype=np.int32)
            indices = np.tile(np.arange(B, dtype=np.int64) % 3, T)
            np.savez(bdir / "b0.npz", lengths=lengths, indices=indices)
            # Should succeed without owner out-of-bounds error
            sp.main([
                "--batches", str(bdir),
                "--out", str(odir),
                "--policy", "greedy",
                "--steps", "1",
                "--gpus", "8",
                "--owner", "1.0",
                "--multihot",
            ])
            with np.load(odir / "b0.npz") as z:
                self.assertEqual(z["gpu"].shape, (B,))
                self.assertEqual(z["rowmax"].shape, (26,))
                np.testing.assert_array_equal(np.bincount(z["gpu"], minlength=8), [2] * 8)


class TestPythonVsCBackendParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls._lib = _compile_placer_for_test(Path(cls._tmpdir.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_python_and_c_backends_match_exactly(self):
        if self._lib is None:
            self.skipTest("No C compiler (gcc or MSVC cl.exe) available to compile placer.c")

        py_be = pl.PythonBackend()
        c_be = pl.CBackend(lib_path=self._lib)
        rng = np.random.default_rng(2026)

        for G in (2, 4, 8):
            B, Tc, nrow = 64, 30, 80
            cap = B // G
            # Include duplicate row ids within a sample (like multi-hot) and invalid slots
            rid = rng.integers(0, nrow, size=(B, Tc), dtype=np.int64)
            valid = rng.random((B, Tc)) > 0.15
            # Force one completely empty sample
            valid[5, :] = False
            w = rng.choice([0.0, 1.0], size=nrow, p=[0.3, 0.7]).astype(np.float32)
            prior = rng.integers(0, 1 << G, size=nrow, dtype=np.uint8)
            bonus = (1 << rng.integers(0, G, size=nrow)).astype(np.uint8)
            bonus[rng.random(nrow) < 0.4] = 0
            owner = rng.integers(0, G, size=nrow, dtype=np.int8)

            for prior_arg in (None, prior):
                for bonus_arg, beta in ((None, 0.0), (bonus, 1.75)):
                    for owner_arg, wown in ((None, 0.0), (owner, 0.85)):
                        g_py, cur_py, dt_py = py_be.assign(
                            rid, valid, G, cap, w, 1.0,
                            prior=prior_arg, bonus=bonus_arg, beta=beta,
                            owner=owner_arg, wown=wown,
                        )
                        g_c, cur_c, dt_c = c_be.assign(
                            rid, valid, G, cap, w, 1.0,
                            prior=prior_arg, bonus=bonus_arg, beta=beta,
                            owner=owner_arg, wown=wown,
                        )
                        self.assertIsNone(dt_py)
                        self.assertIsNotNone(dt_c)
                        np.testing.assert_array_equal(g_py, g_c)
                        np.testing.assert_array_equal(cur_py, cur_c)
                        # Verify cur matches explicit bitwise_or mask
                        expected_mask = pl.compute_gpu_mask(rid, valid, g_c, nrow)
                        np.testing.assert_array_equal(cur_c, expected_mask)


class TestGoldenEndToEndAndConsumers(unittest.TestCase):
    def test_golden_parity_against_original_script_and_step_traffic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orig_script = root / "sample_placement_orig.py"
            if not _extract_original_script(orig_script):
                self.skipTest("git HEAD:analysis/sample_placement.py unavailable")

            bdir = root / "batches"
            _make_synthetic_batches(bdir, steps=5, B=64, T=26, seed=99)

            configs = [
                ["--policy", "contig", "--gpus", "4"],
                ["--policy", "greedy", "--gpus", "8", "--cmax", "16"],
                ["--policy", "refine", "--gpus", "4", "--passes", "2", "--cmax", "16"],
                ["--policy", "affinity", "--gpus", "8", "--passes", "0", "--beta", "2.0", "--lag", "1"],
                ["--policy", "affinity", "--gpus", "4", "--passes", "2", "--beta", "2.0", "--lag", "3"],
                ["--policy", "contig", "--gpus", "4", "--multihot"],
                ["--policy", "affinity", "--gpus", "4", "--passes", "1", "--lag", "2", "--multihot"],
            ]

            for idx, extra in enumerate(configs):
                out_orig = root / f"orig_{idx}"
                out_new = root / f"new_{idx}"
                base_args = ["--batches", str(bdir), "--steps", "5"] + extra

                r_orig = subprocess.run(
                    [sys.executable, str(orig_script), "--out", str(out_orig)] + base_args,
                    capture_output=True, text=True, check=True,
                )
                r_new = subprocess.run(
                    [sys.executable, str(ANALYSIS_DIR / "sample_placement.py"), "--out", str(out_new)] + base_args,
                    capture_output=True, text=True, check=True,
                )

                self.assertEqual(r_new.stdout, r_orig.stdout, f"stdout mismatch for {extra}")
                for step in range(5):
                    with np.load(out_orig / f"b{step}.npz") as zo, np.load(out_new / f"b{step}.npz") as zn:
                        self.assertEqual(set(zn.files), set(zo.files))
                        for key in zo.files:
                            self.assertEqual(zn[key].dtype, zo[key].dtype, f"dtype mismatch {key} {extra}")
                            np.testing.assert_array_equal(
                                zn[key], zo[key],
                                err_msg=f"array mismatch key={key} step={step} args={extra}",
                            )

                # Verify step_traffic.py produces identical CSV on original vs refactored caches
                if "--gpus" in extra:
                    g_val = extra[extra.index("--gpus") + 1]
                    csv_orig = root / f"traffic_orig_{idx}.csv"
                    csv_new = root / f"traffic_new_{idx}.csv"
                    subprocess.run(
                        [sys.executable, str(ANALYSIS_DIR / "step_traffic.py"),
                         str(out_orig), "--steps", "5", "--gpus", g_val, "--out", str(csv_orig)],
                        check=True, capture_output=True, text=True,
                    )
                    subprocess.run(
                        [sys.executable, str(ANALYSIS_DIR / "step_traffic.py"),
                         str(out_new), "--steps", "5", "--gpus", g_val, "--out", str(csv_new)],
                        check=True, capture_output=True, text=True,
                    )
                    self.assertEqual(csv_new.read_text(), csv_orig.read_text())

    def test_backward_compatible_imports_and_linkmodel_integration(self):
        # Verify legacy import surface used by linkmodel.py and placement_check.py
        self.assertEqual(len(sp.MH_SIZES), 26)
        self.assertEqual(sum(sp.MH_SIZES), 214)
        self.assertTrue(callable(sp.multihot))
        self.assertTrue(callable(sp.assign))
        self.assertTrue(callable(sp.load_c))
        self.assertTrue(callable(sp.assign_c))

        import linkmodel

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bdir = root / "batches"
            cdir = root / "cache"
            _make_synthetic_batches(bdir, steps=5, B=32, T=26, seed=11)
            sp.main([
                "--batches", str(bdir),
                "--out", str(cdir),
                "--policy", "affinity",
                "--steps", "5",
                "--gpus", "4",
                "--passes", "1",
            ])

            args = argparse.Namespace(
                batches=bdir, cache=cdir, first=0, steps=3, gpus=4,
                bag_sizes=True, keep_window=2, sample=1, row_shift=26,
            )
            B_cfg = dict(req=64.0, data=576.0, ack=64.0, ctl=64.0)
            flows_p, stats_p = linkmodel.pooled_flows(args, B_cfg, dedup=True, to_pool=False)
            self.assertEqual(len(flows_p), 3)
            self.assertGreater(stats_p["fetch"], 0)

            flows_h, stats_h = linkmodel.hybrid_flows(args, B_cfg, dedup=True, promote=True)
            self.assertEqual(len(flows_h), 3)
            self.assertGreater(stats_h["bag_uses"], 0)


if __name__ == "__main__":
    unittest.main()
