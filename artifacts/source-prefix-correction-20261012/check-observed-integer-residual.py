"""Compare actual operands and block boundaries without changing the reference."""
import gzip
import json
from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


raw = Path(sys.argv[1]).read_bytes()
receipt = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
values = receipt['accumulation']
assert len(values) >= 10336
rows = [values[i * 10:(i + 1) * 10] for i in range(1024)]
counts = dict.fromkeys(['product', 'productResidual', 'sum', 'sumResidual', 'correction'], 0)
for row in rows:
    product, residual, total, error, old, new, previous, a, weight, _ = row
    actual = [product, residual, total, error, new]
    expected = [f32(a * weight), f32(a * weight - product), f32(previous + product),
                f32(previous + product - total), f32(old + f32(residual + error))]
    for key, observed, target in zip(counts, actual, expected):
        counts[key] += observed != target
blocks = []
for block in range(4):
    for lane in range(4):
        indices = [block * 256 + pair * 64 + item * 4 + lane + odd * 32
                   for pair in range(4) for item in range(8) for odd in range(2)]
        terms = [rows[index] for index in indices]
        for previous, current in zip(terms, terms[1:]):
            assert current[6] == previous[2] and current[4] == previous[5]
        exact = sum(row[7] * row[8] for row in terms)
        blocks.append({'block': block, 'lane': lane, 'initialCorrection': terms[0][4],
                       'finalCorrection': terms[-1][5], 'high': terms[-1][2],
                       'exactResidual': exact - terms[-1][2]})
result = {'operationMismatches': counts, 'blocks': blocks,
          'nonzeroBlockInitialCorrections': sum(row['initialCorrection'] != 0 for row in blocks)}
print(json.dumps(result, indent=2))
