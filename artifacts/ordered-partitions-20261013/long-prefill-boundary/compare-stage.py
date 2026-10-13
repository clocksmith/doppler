"""Compare immutable diagnostic GPU copies with independently retained F64 layers."""
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
capture_bytes = (root / 'stage-result.json').read_bytes()
capture = json.loads(capture_bytes)
reference_bytes = (root / 'independent-layers.npz').read_bytes()
binding = json.loads((root / 'independent-layers.json').read_text())
assert hashlib.sha256(reference_bytes).hexdigest() == binding['sha256']
reference = np.load(root / 'independent-layers.npz')
manifest_path = Path('/var/tmp/doppler-cold-compile-model-20261013/manifest.json')
proposal = json.loads((root / 'proposal.json').read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == proposal['files']['manifest']['sha256']
manifest = json.loads(manifest_path.read_text())
entry = manifest['tensors']['model.language_model.layers.16.input_layernorm.weight']
assert entry['dtype'] == 'F16' and entry['shape'] == [1024] and not entry.get('spans')
shard_name = manifest['shards'][entry['shard']]['filename']
shard = (manifest_path.parent / shard_name).read_bytes()
original = json.loads((root / 'result.json').read_text())
assert hashlib.sha256(shard).hexdigest() == next(r['sha256'] for r in original['shards'] if r['path'] == shard_name)
weight = np.frombuffer(shard[entry['offset']:entry['offset'] + entry['size']], dtype='<f2').astype(np.float64)
layer15 = np.load(root / 'independent-layer-15.npy')
assert hashlib.sha256(layer15.astype('<f8').tobytes()).hexdigest() == next(r['independentSha256'] for r in original['boundaries'] if r['layer'] == 15)
# Exact original evaluator equation; retained layer 15 and original F16 weights.
norm16 = layer15 / np.sqrt(np.mean(layer15 * layer15, axis=-1, keepdims=True) + 1e-6) * (weight + 1)

comparisons = []
for row in capture['diagnostic']['snapshots']:
    data = base64.b64decode(row['data'])
    assert row['shape'] == [184, 1024], row['shape']
    assert len(data) == row['byteLength'] == 184 * 1024 * 8
    assert row['storageDigest'] == 'sha256:' + hashlib.sha256(data).hexdigest()
    assert row['stage'] in ('layer.out', 'attn.post_input_norm')
    row = {**row, 'numTokens': 184, 'position': 0}
    actual = np.frombuffer(data, dtype='<f4').astype(np.float64).reshape(row['numTokens'], 1024, 2).sum(axis=-1)
    independent = norm16 if row['stage'] == 'attn.post_input_norm' else reference['layer_' + str(row['layer'])]
    independent = independent[row['position']:row['position'] + row['numTokens']]
    assert actual.shape == independent.shape and np.isfinite(actual).all()
    difference = np.abs(actual - independent)
    comparisons.append({'stage': row['stage'], 'layer': row['layer'], 'position': row['position'], 'tokens': row['numTokens'],
                        'maxAbsoluteError': float(difference.max()),
                        'maxIndex': list(map(int, np.unravel_index(np.argmax(difference), difference.shape))),
                        'rowMaxAbsoluteError': difference.max(axis=1).tolist(),
                        'storageSha256': hashlib.sha256(data).hexdigest()})
result = {'schema': 'doppler.long-prefill-layer-comparison/v1',
          'scope': 'Diagnostic attribution; no new acceptance tolerance or replacement reference',
          'captureSha256': hashlib.sha256(capture_bytes).hexdigest(),
          'referenceSha256': binding['sha256'],
          'unchangedObservedLogits': [row['matchesOriginalBytes'] for row in capture['diagnostic']['cases']],
          'layers': comparisons, 'failure': capture.get('failure'),
          'cleanupFailure': capture.get('cleanupFailure'), 'providerSettlement': capture.get('providerSettlement')}
(root / 'stage-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({**result, 'layers': [{k: v for k, v in row.items() if k != 'rowMaxAbsoluteError'}
                                   for row in comparisons]}, indent=2))
