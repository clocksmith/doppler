"""Validate and compare canonical layer-17 observations, retaining duplicates."""
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
archive = (root / 'layer17-captures.json.gz').read_bytes()
receipt = json.loads((root / 'layer17-result.json').read_text())
assert hashlib.sha256(archive).hexdigest() == receipt['diagnostic']['export']['sha256']
reference_bytes = (root / 'reference-layer17.npz').read_bytes()
reference_binding = json.loads((root / 'reference-layer17.json').read_text())
assert hashlib.sha256(reference_bytes).hexdigest() == reference_binding['sha256']
reference = np.load(root / 'reference-layer17.npz')
records = json.loads(gzip.decompress(archive))['records']
rows = []
for row in records:
    assert row['layer'] == 17
    expected = reference[row['stage']]
    assert row['shape'] == list(expected.shape)
    data = base64.b64decode(row['data'])
    assert len(data) == row['byteLength'] == expected.size * 8
    assert row['storageDigest'] == 'sha256:' + hashlib.sha256(data).hexdigest()
    actual = np.frombuffer(data, dtype='<f4').astype(np.float64).reshape(*expected.shape, 2).sum(axis=-1)
    assert np.isfinite(actual).all()
    error = np.abs(actual - expected)
    rows.append({'stage': row['stage'], 'ordinal': row['ordinal'], 'shape': row['shape'],
                 'maxAbsoluteError': float(error.max()),
                 'maxIndex': list(map(int,np.unravel_index(error.argmax(), error.shape))),
                 'zeroRows': int(np.sum(np.all(actual == 0, axis=-1))),
                 'firstDivergentRow': next((i for i, v in enumerate(error.max(axis=-1)) if v > 2e-11), None),
                 'storageDigest': row['storageDigest']})
result = {'scope': 'Attribution only; unchanged numerical acceptance reference',
          'captureSha256': hashlib.sha256(archive).hexdigest(), 'reference': reference_binding,
          'observedCases': [{k:v for k,v in row.items() if k != 'logits'} for row in receipt['diagnostic']['cases']],
          'records': rows, 'failure': receipt.get('failure'), 'cleanupFailure': receipt.get('cleanupFailure')}
(root / 'layer17-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
