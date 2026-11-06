"""Compare a receipt-bound GPU observation against the independent artifact/KV control.

Uses the existing quantized-v1 policy unchanged. A passing sampled layer slice
is not full-tensor parity. Full last-position logits are checked when retained.
Usage: python3 -P compare-learned-reference.py <gpu.json> <reference.json> <output.json>
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

gpu_path, reference_path, output_path = map(Path, sys.argv[1:])
gpu_bytes, reference_bytes = gpu_path.read_bytes(), reference_path.read_bytes()
gpu, reference = json.loads(gpu_bytes), json.loads(reference_bytes)
policy_path = Path(__file__).resolve().parents[2] / 'src/config/evidence/boundary-evidence-policy.json'
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = json.loads(policy_path.read_bytes())['tolerancePolicies'][policy_id]
assert gpu.get('completed') and not gpu.get('failure')
assert not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert reference['artifactManifestSha256'] == gpu['manifestSha256']
assert reference['kvStorageDtype'] == 'f16'

def compare(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape and actual.size
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    error = np.abs(actual - expected)
    limit = policy['absolute'] + policy['relative'] * np.abs(expected)
    return {'count': actual.size, 'maxAbsoluteError': float(error.max()),
            'rmsError': float(np.sqrt(np.mean(error * error))),
            'outsidePolicy': int(np.sum(error > limit)), 'passed': bool(np.all(error <= limit))}

rows = []
for row in gpu['rows']:
    expected = reference['sequences'][row['id']]
    assert row['inputTokenCount'] == expected['tokens']
    position = expected['tokens'] - 1
    result = {'id': row['id'], 'tokenIds': row['tokenIds'],
              'firstTokenAgrees': row['tokenId'] == expected['argmax'][position], 'layers': []}
    for layer in row.get('layers', []):
        assert [r['position'] for r in layer['rows']] == expected['samplePositions']
        values = [r['observed'] for r in layer['rows']]
        result['layers'].append({'layer': layer['layer'], 'scope': 'sampled stream coordinates',
                                 **compare(values, expected['layers'][str(layer['layer'])]['samples'])})
    if row.get('prefillLogitsFile'):
        raw = Path(row['prefillLogitsFile']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['prefillLogitsSha256']
        saved = reference_path.parent / expected['logitsFile']
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected['logitsSha256']
        expected_logits = np.load(saved, allow_pickle=False)[expected['samplePositions'].index(position)]
        result['logits'] = compare(np.frombuffer(raw, dtype='<f4'), expected_logits)
    result['passed'] = result['firstTokenAgrees'] and bool(result['layers'] or 'logits' in result) and all(l['passed'] for l in result['layers'])
    if 'logits' in result:
        result['passed'] = result['passed'] and result['logits']['passed']
    result['fullLogitsChecked'] = 'logits' in result
    rows.append(result)

result = {'schema': 'doppler.qwen38-learned-comparison/v1',
          'scope': 'Named unpadded Node observations only; sampled layers and retained last-position logits, no general model or performance qualification',
          'gpuSha256': hashlib.sha256(gpu_bytes).hexdigest(),
          'referenceSha256': hashlib.sha256(reference_bytes).hexdigest(),
          'policySha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
          'policyId': policy_id, 'policy': policy, 'rows': rows,
          'passed': bool(rows) and all(row['passed'] for row in rows)}
output_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'passed': result['passed'], 'rows': len(rows), 'output': str(output_path)}))
if not result['passed']:
    raise SystemExit(1)
