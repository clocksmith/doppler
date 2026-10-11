#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[2] / 'tools/qwen35-final-projection-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProjectionDecomposition(unittest.TestCase):
    def test_inherited_error_is_not_assigned_to_final_projection(self):
        operands = np.array([2 ** 24, 1], dtype=np.float32)
        weight = np.array([[1, 1], [1, -1]], dtype=np.float32)
        actual = np.array([2 ** 24, 2 ** 24 - 1], dtype=np.float32)
        independent = np.array([2 ** 24 + 2, 2 ** 24 - 2], dtype=np.float64)
        _, local, inherited = audit.decompose(actual, operands, weight, independent)
        np.testing.assert_array_equal(local, [-1, 0])
        np.testing.assert_array_equal(inherited, [-1, 1])

    def test_nonfinite_operands_cannot_certify_accuracy(self):
        with self.assertRaises(AssertionError):
            audit.decompose(np.array([1]), np.array([np.nan]), np.array([[1]]), np.array([1]))


if __name__ == '__main__':
    unittest.main()
