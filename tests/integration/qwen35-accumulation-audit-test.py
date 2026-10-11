#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[2] / 'tools/qwen35-accumulation-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Accumulation(unittest.TestCase):
    def test_tile_subtotals_recover_small_contributions(self):
        x = np.ones((1, 1024), dtype=np.float32)
        x[0, 0] = 2 ** 26
        w = np.ones((1, 1024), dtype=np.float32)
        precise = x.astype(np.float64) @ w.astype(np.float64).T
        ordinary = audit.accumulate(x, w, 'fma', 1)
        blocked = audit.accumulate(x, w, 'blocked-tile', 16)
        self.assertEqual(float(abs(ordinary - precise)[0, 0]), 1023)
        self.assertLess(float(abs(blocked - precise)[0, 0]), 20)
        self.assertEqual(blocked.dtype, np.float32)

    def test_signed_ragged_shapes(self):
        x = np.arange(-34, 35, dtype=np.float32).reshape(3, 23) / 16
        w = np.arange(-57, 58, dtype=np.float32).reshape(5, 23) / 32
        precise = x.astype(np.float64) @ w.astype(np.float64).T
        actual = audit.accumulate(x, w, 'blocked-tile', 16)
        np.testing.assert_array_equal(actual, precise)

    def test_q4_block_partials_shorten_four_lane_chains(self):
        x = np.ones((1, 1024), dtype=np.float32)
        x[0, 0] = 2 ** 26
        weight = np.ones_like(x)
        precise = x.astype(np.float64) @ weight.astype(np.float64).T
        ordinary = audit.accumulate(x, weight, 'q4-widetile')
        blocked = audit.accumulate(x, weight, 'q4-widetile-blocked')
        self.assertLess(float(abs(blocked - precise)[0, 0]), float(abs(ordinary - precise)[0, 0]))
        # One quantization block must retain the exact original FMA order.
        np.testing.assert_array_equal(audit.accumulate(x[:, :256], weight[:, :256], 'q4-widetile'),
                                      audit.accumulate(x[:, :256], weight[:, :256], 'q4-widetile-blocked'))
        with self.assertRaises(AssertionError):
            audit.accumulate(x[:, :257], weight[:, :257], 'q4-widetile')

    def test_invalid_operands_reject(self):
        with self.assertRaises(AssertionError):
            audit.accumulate(np.ones((2, 3)), np.ones((2, 3)), 'blocked-tile', 16)
        with self.assertRaises(AssertionError):
            audit.accumulate(np.ones((2, 3), dtype=np.float32), np.ones((2, 4), dtype=np.float32), 'blocked-tile', 16)


if __name__ == '__main__':
    unittest.main()
