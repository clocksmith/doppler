"""Regression checks for scale-aware evidence, not model acceptance."""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('stage_metrics', Path(__file__).with_name('compare-upstream-stages.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class StageMetricsTests(unittest.TestCase):
    def test_common_scaling_preserves_relative_error(self):
        reference = np.array([2, 4], dtype=np.float32)
        actual = reference + np.float32(0.25)
        before = metrics.error_metrics(actual, reference)
        after = metrics.error_metrics(actual * 128, reference * 128)
        self.assertEqual(after['maxAbsoluteDifference'], before['maxAbsoluteDifference'] * 128)
        self.assertEqual(after['referenceRms'], before['referenceRms'] * 128)
        self.assertEqual(after['maxErrorOverReferenceRms'], before['maxErrorOverReferenceRms'])
        self.assertEqual(after['rmsErrorOverReferenceRms'], before['rmsErrorOverReferenceRms'])

    def test_global_rms_does_not_hide_small_stream(self):
        reference = np.array([[0.001, 0.001, 1, 1]], dtype=np.float32)
        actual = reference + np.float32(0.001)
        row = metrics.stage_metrics(actual, reference, [2135], 2, 2)
        streams = row['positionMetrics'][0]['streams']
        self.assertGreater(streams[0]['maxErrorOverReferenceRms'], 0.99)
        self.assertLess(streams[1]['maxErrorOverReferenceRms'], 0.002)
        self.assertLess(row['maxErrorOverReferenceRms'], 0.002)

    def test_zero_scale_is_explicit_and_json_safe(self):
        reference = np.zeros(4, dtype=np.float32)
        equal = metrics.error_metrics(reference, reference)
        self.assertEqual(equal['maxErrorOverReferenceRms'], 0)
        changed = metrics.error_metrics(np.ones(4, dtype=np.float32), reference)
        self.assertIsNone(changed['maxErrorOverReferenceRms'])
        self.assertFalse(changed['relativeErrorDefined'])
        json.dumps(changed, allow_nan=False)

    def test_missing_geometry_or_invalid_values_do_not_become_metrics(self):
        x = np.ones((1, 4), dtype=np.float32)
        with self.assertRaises(AssertionError):
            metrics.stage_metrics(x, x, [0], 3, 2)
        with self.assertRaises(AssertionError):
            metrics.error_metrics(np.array([np.nan]), np.array([1.0]))


if __name__ == '__main__':
    unittest.main()
