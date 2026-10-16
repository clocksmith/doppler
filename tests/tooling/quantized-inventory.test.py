import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'inventory', Path(__file__).resolve().parents[2] / 'tools/inventory-quantized-model.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class TensorSpans(unittest.TestCase):
    def test_exact_payload_and_cross_shard_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').write_bytes(b'xxABCyy')
            (root / 'b').write_bytes(b'zzDEFG')
            manifest = {'shards': [{'filename': 'a'}, {'filename': 'b'}]}
            entry = {'size': 7, 'spans': [
                {'shardIndex': 0, 'offset': 2, 'size': 3},
                {'shardIndex': 1, 'offset': 2, 'size': 4}]}
            read = lambda start=0, size=None: b''.join(inventory.tensor_chunks(root, manifest, entry, start, size))
            self.assertEqual(read(), b'ABCDEFG')
            self.assertEqual(read(2, 3), b'CDE')
            self.assertEqual(read(3, 4), b'DEFG')
            self.assertEqual(read(7, 0), b'')
            with self.assertRaises(AssertionError):
                read(6, 2)
            (root / 'b').write_bytes(b'zzDE')
            with self.assertRaisesRegex(AssertionError, 'Truncated'):
                read()

    def test_single_shard_and_missing_span(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').write_bytes(b'0123456')
            manifest = {'shards': [{'filename': 'a'}]}
            entry = {'shard': 0, 'offset': 2, 'size': 3}
            self.assertEqual(b''.join(inventory.tensor_chunks(root, manifest, entry)), b'234')
            entry = {'size': 4, 'spans': [{'shardIndex': 0, 'offset': 2, 'size': 3}]}
            with self.assertRaisesRegex(AssertionError, 'Incomplete'):
                b''.join(inventory.tensor_chunks(root, manifest, entry))


if __name__ == '__main__':
    unittest.main()
