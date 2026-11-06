"""Compare a receipt-bound GPU observation against the independent artifact/KV control.

Uses the existing quantized-v1 policy unchanged. A passing sampled layer slice
is not full-tensor parity. Full last-position logits are checked when retained.
Usage: python3 -P compare-learned-reference.py <gpu.json> <reference.json> <output.json>
       [<decode-reference.json> <decode-inputs.json>]
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

assert len(sys.argv) in (4, 6)
gpu_path, reference_path, output_path = map(Path, sys.argv[1:4])
gpu_bytes, reference_bytes = gpu_path.read_bytes(), reference_path.read_bytes()
gpu, reference = json.loads(gpu_bytes), json.loads(reference_bytes)
policy_path = Path(__file__).resolve().parents[2] / 'src/config/evidence/boundary-evidence-policy.json'
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = json.loads(policy_path.read_bytes())['tolerancePolicies'][policy_id]
assert gpu.get('completed') and not gpu.get('failure')
assert not gpu.get('routingControl'), 'counterfactual router replacement cannot qualify normal model execution'
assert not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert reference['artifactManifestSha256'] == gpu['manifestSha256']
assert reference['inputs'] == 'learned-forward-inputs.json', 'this comparison requires the named frozen reference inputs'
original_bytes = (Path(__file__).parent / reference['inputs']).read_bytes()
assert gpu['inputsSha256'] == hashlib.sha256(original_bytes).hexdigest(), 'GPU observation must use the same frozen inputs'
original_inputs = {row['id']: row['tokens'] for row in json.loads(original_bytes)['sequences']}
assert reference['kvStorageDtype'] == 'f16'
decode_reference = None
if len(sys.argv) == 6:
    decode_path, decode_inputs_path = map(Path, sys.argv[4:])
    decode_bytes, decode_input_bytes = decode_path.read_bytes(), decode_inputs_path.read_bytes()
    decode_reference = json.loads(decode_bytes)
    assert decode_reference['artifactManifestSha256'] == gpu['manifestSha256']
    assert decode_reference['kvStorageDtype'] == 'f16'
    assert decode_reference['inputsSha256'] == hashlib.sha256(decode_input_bytes).hexdigest()
    decode_inputs = {row['id']: row['tokens'] for row in json.loads(decode_input_bytes)['sequences']}

def compare(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape and actual.size
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    error = np.abs(actual - expected)
    limit = policy['absolute'] + policy['relative'] * np.abs(expected)
    return {'count': actual.size, 'maxAbsoluteError': float(error.max()),
            'rmsError': float(np.sqrt(np.mean(error * error))),
            'outsidePolicy': int(np.sum(error > limit)), 'passed': bool(np.all(error <= limit))}

def full_logits(file, digest, expected, directory, position):
    raw = Path(file).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    saved = directory / expected['logitsFile']
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected['logitsSha256']
    expected_logits = np.load(saved, allow_pickle=False)[expected['samplePositions'].index(position)]
    return compare(np.frombuffer(raw, dtype='<f4'), expected_logits)

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
        result['logits'] = full_logits(row['prefillLogitsFile'], row['prefillLogitsSha256'], expected, reference_path.parent, position)
    result['passed'] = result['firstTokenAgrees'] and bool(result['layers'] or 'logits' in result) and all(l['passed'] for l in result['layers'])
    if 'logits' in result:
        result['passed'] = result['passed'] and result['logits']['passed']
    result['fullLogitsChecked'] = 'logits' in result
    if decode_reference:
        result['decodeRows'] = []
        assert len(row['stepLogits']) == len(row['tokenIds'])
        for observed in row['stepLogits'][1:]:
            step = observed['step']
            key = row['id'] + ('-next' if step == 1 else f'-next{step}')
            expected = decode_reference['sequences'][key]
            context = original_inputs[row['id']] + row['tokenIds'][:step]
            assert context == decode_inputs[key], 'independent decode control must use the exact generated prefix'
            assert len(context) == expected['tokens'] == observed['inputTokenCount']
            comparison = full_logits(observed['file'], observed['sha256'], expected, decode_path.parent, len(context) - 1)
            agrees = observed['tokenId'] == expected['argmax'][-1]
            comparison.update(step=step, tokenId=observed['tokenId'], tokenAgrees=agrees)
            comparison['passed'] = comparison['passed'] and agrees
            result['decodeRows'].append(comparison)
        result['passed'] = result['passed'] and all(d['passed'] for d in result['decodeRows'])
    rows.append(result)

result = {'schema': 'doppler.qwen38-learned-comparison/v1',
          'scope': 'Named unpadded Node observations only; sampled layers and retained last-position logits, no general model or performance qualification',
          'gpuSha256': hashlib.sha256(gpu_bytes).hexdigest(),
          'referenceSha256': hashlib.sha256(reference_bytes).hexdigest(),
          'policySha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
          'policyId': policy_id, 'policy': policy, 'rows': rows,
          'passed': bool(rows) and all(row['passed'] for row in rows)}
if decode_reference:
    result['decodeReferenceSha256'] = hashlib.sha256(decode_bytes).hexdigest()
    result['decodeInputsSha256'] = hashlib.sha256(decode_input_bytes).hexdigest()
output_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'passed': result['passed'], 'rows': len(rows), 'output': str(output_path)}))
if not result['passed']:
    raise SystemExit(1)
