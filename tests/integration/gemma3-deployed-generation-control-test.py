"""Closed-form stored-weight decoding controls; not physical model acceptance."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'gemma_control', Path(__file__).parents[2] / 'tools/gemma3-deployed-generation-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)

def decode(data, dtype, shape):
    return control.decode_weight(data, {'dtype': dtype, 'shape': shape,
                                       'size': len(data), 'layout': 'row'}).numpy()

for dtype, storage in [('F16', '<f2'), ('F32', '<f4')]:
    values = np.array([1, -2, 0, 0.5], dtype=storage)
    np.testing.assert_array_equal(decode(values.tobytes(), dtype, [2, 2]), values.reshape(2, 2))
bf16 = np.array([0x3f80, 0xc000, 0, 0x3f00], dtype='<u2')
np.testing.assert_array_equal(decode(bf16.tobytes(), 'BF16', [4]), [1, -2, 0, 0.5])

# Two rows with a partly used second block each. Every block is constant,
# making row padding distinguishable from flat-stream truncation.
blocks = []
for value in [1, 2, 3, 4]:
    block = bytearray(144)
    block[:2] = np.array([value], dtype='<f2').tobytes()
    block[4:16] = bytes([1]) * 12
    block[16:] = bytes([0x11]) * 128
    blocks.append(block)
packed = bytes().join(blocks)
actual = decode(packed, 'Q4_K_M', [2, 257])
np.testing.assert_array_equal(actual[0, :256], 1)
np.testing.assert_array_equal(actual[1, :256], 3)
np.testing.assert_array_equal(actual[:, 256], [2, 4])
try:
    decode(packed[:-144], 'Q4_K_M', [2, 257])
    raise RuntimeError('Truncated row padding was accepted')
except ValueError as error:
    assert 'row storage' in str(error)
print('Five stored-weight decoding controls passed; no model qualification claimed.')
