"""Cross-check offline model arithmetic against the independent BigInt oracle."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from lib.w8a8_model_reference import quantize_rows, rescale, decode_rows, IndependentW8A8Linear


class W8A8ModelReferenceTest(unittest.TestCase):
    def test_quantization_matches_bigint_oracle(self):
        rng = np.random.default_rng(7331)
        rows = (rng.uniform(-1, 1, (13, 257)) * np.exp2(np.arange(-60, 70, 10))[:, None]).astype(np.float32)
        rows[0] = 0
        rows[1] = 0
        rows[1, :2] = [0.001003009034320712, 0.00001184656321129296]
        request = {'values': rows.flatten().tolist(), 'rows': len(rows), 'k': rows.shape[1]}
        script = """
import { quantizeW8A8Reference } from './tests/kernels/reference/w8a8.js';
const chunks = []; for await (const chunk of process.stdin) chunks.push(chunk);
const input = JSON.parse(chunks.join(''));
const result = quantizeW8A8Reference(new Float32Array(input.values), input.rows, input.k);
process.stdout.write(JSON.stringify({codes:[...result.codes],scales:[...result.scales]}));
"""
        result = subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(request), text=True, capture_output=True, check=True)
        expected = json.loads(result.stdout)
        codes, scales = quantize_rows(rows)
        np.testing.assert_array_equal(codes, np.array(expected['codes']).reshape(len(rows), -1)[:, :rows.shape[1]])
        np.testing.assert_array_equal(scales, expected['scales'])
        self.assertEqual(codes[1, 1], 1)

    def test_integer_matmul_and_separate_rounding(self):
        codes = np.full((1, 133144), 127, np.int8)
        module = IndependentW8A8Linear(codes, np.ones(1, np.float32))
        result = module(torch.full((1, 133144), 127.0))
        self.assertEqual(result.item(), np.float32(2147479576))
        result = rescale(torch.tensor([[-1]], dtype=torch.int32), torch.tensor([2.0 ** -126]), torch.tensor([0.5]))
        self.assertEqual(result.item(), 0)
        self.assertTrue(torch.signbit(result).item())
        with self.assertRaisesRegex(ValueError, 'overflow'):
            rescale(torch.tensor([[2147483647]], dtype=torch.int32), torch.tensor([1e30]), torch.tensor([1.0]))

    def test_rejected_source_and_scale(self):
        for value in [float('nan'), float('inf'), np.nextafter(np.float32(0), np.float32(1)), 2 ** -126]:
            with self.assertRaises(ValueError):
                quantize_rows(np.array([[value]], np.float32))
        boundary = np.float32(127 * 2 ** -126)
        below = np.nextafter(boundary, np.float32(0))
        self.assertLess(np.float32(float(below) / 127), np.float32(2 ** -126))
        with self.assertRaises(ValueError):
            quantize_rows(np.array([[below]], np.float32))
        self.assertEqual(quantize_rows(np.array([[boundary]], np.float32))[1][0], np.float32(2 ** -126))


if __name__ == '__main__':
    unittest.main()
