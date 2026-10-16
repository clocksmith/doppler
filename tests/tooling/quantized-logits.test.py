import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'comparison', Path(__file__).resolve().parents[2] / 'tools/compare-quantized-logits.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class LogitAcceptance(unittest.TestCase):
    def test_npy_parsing_checks_layout_and_payload_length(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'logits.npy'
            for kind, tag in [('f', '<f4'), ('d', '<f8')]:
                header = repr(dict(descr=tag, fortran_order=False, shape=(1, 2))).encode() + b'\n'
                payload = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<2' + kind, 1, -2)
                filename.write_bytes(payload)
                self.assertEqual(comparison.read_npy(filename), [(1.0, -2.0)])
                filename.write_bytes(payload[:-1])
                with self.assertRaises(struct.error):
                    comparison.read_npy(filename)

    def test_near_tie_changes_token_without_failing_numerical_bound(self):
        row = comparison.compare([1.0, 0.9999, -2.0], [0.9999, 1.0, -2.0], 0.001)
        self.assertTrue(row['passed'])
        self.assertNotEqual(row['referenceTopToken'], row['dopplerTopToken'])

    def test_same_token_does_not_hide_numerical_error(self):
        row = comparison.compare([1.0, 0.5, -2.0], [1.0, 0.6, -2.0], 0.001)
        self.assertFalse(row['passed'])
        self.assertEqual(row['referenceTopToken'], row['dopplerTopToken'])

    def test_rms_bound_detects_widespread_error_below_maximum_bound(self):
        row = comparison.compare([1.0, 0.5, -2.0], [1.01, 0.51, -1.99], 0.02, 0.001)
        self.assertLess(row['maxAbsoluteError'], 0.02)
        self.assertFalse(row['passed'])
        with self.assertRaises(AssertionError):
            comparison.compare([1.0, 2.0], [1.0, 2.0], 0.02, math.nan)

    def test_invalid_vectors_and_bounds_fail(self):
        for a, b, bound in [([], [], 0.1), ([1], [1], 0.1), ([1, 2], [1], 0.1),
                            ([1, math.nan], [1, 2], 0.1), ([1, 2], [1, math.inf], 0.1),
                            ([1, 2], [1, 2], -1), ([1, 2], [1, 2], math.inf)]:
            with self.subTest(a=a, b=b, bound=bound), self.assertRaises(AssertionError):
                comparison.compare(a, b, bound)


if __name__ == '__main__':
    unittest.main()
