"""Corrupt observation bytes must fail before producing numerical evidence."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[2] / 'reports/flash-next-qualification-20261106/verified_tensor_io.py'
spec = importlib.util.spec_from_file_location('verified_tensor_io', MODULE)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class TensorIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = np.array([[1, 2, 3]], dtype=np.float32)

    def record(self, suffix):
        file = self.root / ('tensor' + suffix)
        if suffix == '.npy':
            np.save(file, self.values, allow_pickle=False)
        else:
            file.write_bytes(self.values.tobytes())
        return dict(file=file.name, shape=list(self.values.shape),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest())

    def test_valid_encodings(self):
        for suffix in ('.f32', '.npy'):
            np.testing.assert_array_equal(reader.read_tensor(self.record(suffix), self.root), self.values)

    def test_changed_payload_fails_with_unchanged_receipt(self):
        for suffix in ('.f32', '.npy'):
            record = self.record(suffix)
            file = self.root / record['file']
            data = bytearray(file.read_bytes())
            data[-1] ^= 1
            file.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                reader.read_tensor(record, self.root)

    def test_wrong_shape_fails_for_both_encodings(self):
        for suffix in ('.f32', '.npy'):
            record = self.record(suffix)
            record['shape'] = [1, 2]
            with self.assertRaisesRegex(ValueError, 'shape'):
                reader.read_tensor(record, self.root)

    def test_missing_hash_and_nonfinite_fail(self):
        record = self.record('.f32')
        del record['sha256']
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            reader.read_tensor(record, self.root)
        self.values[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            reader.read_tensor(self.record('.f32'), self.root)

    def test_trace_rehashes_payloads_and_rejects_duplicates(self):
        record = dict(self.record('.f32'), role='attention', stage='input')
        file = self.root / 'trace.json'
        file.write_text(json.dumps(dict(tensors=[record])))
        _, _, proof = reader.load_verified_trace(file, ('role', 'stage'))
        self.assertEqual(proof['tensors'], 1)
        file.write_text(json.dumps(dict(tensors=[record, record])))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            reader.load_verified_trace(file, ('role', 'stage'))
        file.write_text(json.dumps(dict(tensors=[record])))
        (self.root / record['file']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            reader.load_verified_trace(file, ('role', 'stage'))

    def test_routing_indices_are_read_without_float_conversion(self):
        file = self.root / 'indices.u32'
        values = np.array([[0, 511]], dtype='<u4')
        file.write_bytes(values.tobytes())
        record = dict(file=file.name, shape=[1, 2], sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        actual = reader.read_tensor(record, self.root)
        self.assertEqual(actual.dtype, values.dtype)
        np.testing.assert_array_equal(actual, values)

    def test_both_analysis_entrypoints_reject_corruption_before_output(self):
        record = dict(self.record('.f32'), layer=0, role='attn_hc', stage='input')
        (self.root / record['file']).write_bytes(b'corrupted')
        for script, trace in [('bind-decompose-layer6.py', 'linear-layer6-stages-20261106.json'),
                              ('estimate-injection-steps.py', 'relative-prefix-gpu-20261106.json')]:
            (self.root / trace).write_text(json.dumps(dict(tensors=[record])))
            output = self.root / 'must-not-exist.json'
            result = subprocess.run([sys.executable, '-B', '-O', '-P', str(MODULE.with_name(script)),
                                     str(self.root), str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Tensor SHA-256 mismatch', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
