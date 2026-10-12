"""Decompose captured F32 normalization error against unchanged independent F64."""
import base64
import json
from pathlib import Path
import sys
import numpy as np

fixture = json.loads(Path(sys.argv[1]).read_bytes())
decode = lambda name, dtype: np.frombuffer(base64.b64decode(fixture[name]), dtype=dtype).astype(np.float64)
x = decode('input', '<f4').reshape(fixture['rows'], fixture['width'])
w = 1 + decode('weight', '<f2')
expected = decode('expected', '<f8').reshape(x.shape)
actual = decode('observed', '<f4').reshape(x.shape)
f = lambda values: values.astype(np.float32).astype(np.float64)
mean = (x * x).mean(-1, keepdims=True)
inverse = 1 / np.sqrt(mean + fixture['epsilon'])
groups = x.reshape(len(x), 256, 4)
partial = np.zeros((len(x), 256))
for index in range(4):
    partial = f(partial + f(groups[:, :, index] ** 2))
for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
    partial[:, :stride] = f(partial[:, :stride] + partial[:, stride:2 * stride])
scale = f(1 / np.sqrt(f(partial[:, :1] / 1024 + np.float32(fixture['epsilon']))))
variants = {'actual': actual, 'roundedOracle': f(expected),
    'roundedScaleAndTwoMultiplies': f(f(x * f(inverse)) * w),
    'roundedMeanScaleAndTwoMultiplies': f(f(x * f(1 / np.sqrt(f(mean + fixture['epsilon'])))) * w),
    'productionTreeNearestScale': f(f(x * scale) * w)}
assert np.array_equal(variants['productionTreeNearestScale'], actual)
result = {'scope': 'Captured-input arithmetic decomposition; no new acceptance reference',
          'incorrectScaleRows': int(np.count_nonzero(scale != f(inverse))), 'rows': len(x),
          'values': actual.size, 'variants': {}}
for name, values in variants.items():
    result['variants'][name] = {'maximum': float(np.max(abs(values - expected))),
        'rms': float(np.sqrt(np.mean((values - expected) ** 2))),
        'sameGPU': int(np.count_nonzero(values == actual))}
print(json.dumps(result, indent=2))
