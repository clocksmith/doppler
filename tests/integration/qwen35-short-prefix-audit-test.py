#!/usr/bin/env python3
"""Capture pairing checks; no model or physical qualification."""
import base64
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('short_audit',
    Path(__file__).parents[2] / 'tools/qwen35-short-prefix-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(name, count, value):
    data = np.full((count, 2), value, dtype='<f4')
    return {'opId': name, 'capture': {'level': 'full', 'encoding': 'base64-f32le',
        'shape': list(data.shape), 'dataBase64': base64.b64encode(data.tobytes()).decode()}}


class Pairing(unittest.TestCase):
    def test_equal_maxima_do_not_imply_same_error_source(self):
        expected = np.zeros((1, 4))
        actual = np.array([[.002, -.002, 0., 0.]])
        opposite = -actual
        result = audit.compare_error_sources(actual, opposite, expected)
        self.assertEqual(result['tailOverlap'], 2)
        self.assertEqual(result['sameSignTailOverlap'], 0)
        self.assertAlmostEqual(result['signedErrorCorrelation'], -1.)
        self.assertAlmostEqual(result['actualAgainstControlled']['max'], .004)
        disjoint = np.array([[0., 0., .002, -.002]])
        result = audit.compare_error_sources(actual, disjoint, expected)
        self.assertEqual(result['tailOverlap'], 0)
        self.assertAlmostEqual(result['signedErrorCosine'], 0.)

    def test_ideal_f32_can_double_round_before_half_store(self):
        midpoint = 1.0 + 2.0 ** -11
        values = np.array([[midpoint + 2.0 ** -26, midpoint - 2.0 ** -26, 1.0]], dtype=np.float64)
        result = audit.cache_double_rounding(values)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['coordinates'][0]['index'], [0, 0])
        self.assertEqual(result['coordinates'][0]['nearestF32'], midpoint)
        self.assertEqual(result['coordinates'][0]['throughF32F16'], 1.0)
        self.assertEqual(result['coordinates'][0]['directF16'], 1.0 + 2.0 ** -10)

    def capture(self):
        return {'observation': {'timeline': [row('embed.out', 19, 1),
            row('layer.0.layer.out', 19, 2), row('layer.0.layer.out', 19, 2),
            row('layer.0.ffn.gate', 19, 3), row('embed.out', 1, 4),
            row('layer.0.layer.out', 1, 5)]}}

    def test_pair_by_step_and_preserve_decode(self):
        result = audit.read_capture(self.capture())
        np.testing.assert_array_equal(result['layer.0.layer.out'][:, 0], [2] * 19 + [5])
        self.assertNotIn('layer.0.ffn.gate', result)

    def test_longer_prefix_pairs_every_decode_step(self):
        timeline = []
        for step, count in enumerate([184, 1, 1, 1]):
            timeline.extend([row('embed.out', count, step),
                             row('layer.0.layer.out', count, step + 10)])
        result = audit.read_capture({'observation': {'timeline': timeline}}, 184, 3)
        np.testing.assert_array_equal(result['layer.0.layer.out'][:, 0],
                                      [10] * 184 + [11, 12, 13])
        del timeline[5]
        with self.assertRaises(AssertionError):
            audit.read_capture({'observation': {'timeline': timeline}}, 184, 3)

    def test_conflicting_duplicate_rejected(self):
        capture = self.capture()
        capture['observation']['timeline'][2] = row('layer.0.layer.out', 19, 9)
        with self.assertRaises(AssertionError):
            audit.read_capture(capture)

    def test_missing_decode_not_silently_replaced_with_prefill(self):
        capture = self.capture()
        capture['observation']['timeline'].pop()
        with self.assertRaises(AssertionError):
            audit.read_capture(capture)

    def test_token_count_mismatch_rejected(self):
        capture = self.capture()
        capture['observation']['timeline'][-1] = row('layer.0.layer.out', 2, 5)
        with self.assertRaises(AssertionError):
            audit.read_capture(capture)

    def test_distribution_keeps_tail_and_nonfinite_failure(self):
        result = audit.error(np.array([[0., .0001], [.0009, .002]]), np.zeros((2, 2)))
        self.assertEqual(result['overTolerance'], 1)
        self.assertEqual(result['lastTokenMax'], .002)
        with self.assertRaises(AssertionError):
            audit.error(np.array([np.nan]), np.zeros(1))


if __name__ == '__main__':
    unittest.main()
