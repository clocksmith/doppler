"""Compare a receipt-bound GPU observation against the independent artifact/KV control.

Uses the existing quantized-v1 policy unchanged. A passing sampled layer slice
is not full-tensor parity. Full last-position logits are checked when retained.
Usage: python3 -P compare-learned-reference.py <gpu.json> <reference.json> <output.json>
       [<decode-reference.json> <decode-inputs.json>]
       [--kv-dtype f16|f32] [--inputs <original-inputs.json>]
       [--protocol <decode-protocol.json>] [--policy <policy.json>]
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError(message)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('gpu', type=Path)
parser.add_argument('reference', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('decode_reference', type=Path, nargs='?')
parser.add_argument('decode_inputs', type=Path, nargs='?')
parser.add_argument('--kv-dtype', choices=['f16', 'f32'], default='f16')
parser.add_argument('--inputs', type=Path)
parser.add_argument('--policy', type=Path)
parser.add_argument('--protocol', type=Path)
args = parser.parse_args()
gpu_path, reference_path, output_path = args.gpu, args.reference, args.output
gpu_bytes, reference_bytes = gpu_path.read_bytes(), reference_path.read_bytes()
gpu, reference = json.loads(gpu_bytes), json.loads(reference_bytes)
policy_path = args.policy or Path(__file__).resolve().parents[2] / 'src/config/evidence/boundary-evidence-policy.json'
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy_bytes = policy_path.read_bytes()
policy = json.loads(policy_bytes)['tolerancePolicies'][policy_id]
require(gpu.get('completed') and not gpu.get('failure'), 'GPU observation did not complete')
require(not gpu.get('routingControl') and not gpu.get('counterfactual'), 'Substituted execution cannot qualify')
require(not gpu['gpuErrors'] and not gpu['cleanupFailures'], 'GPU or cleanup errors remain')
require(reference['artifactManifestSha256'] == gpu['manifestSha256'], 'Artifact identity mismatch')
require(gpu['referenceSha256'] == hashlib.sha256(reference_bytes).hexdigest(), 'GPU/reference binding mismatch')
if args.inputs is None:
    require(reference['inputs'] == 'learned-forward-inputs.json', 'Non-default inputs require --inputs')
original_path = args.inputs or Path(__file__).parent / reference['inputs']
original_bytes = original_path.read_bytes()
original_inputs_sha = hashlib.sha256(original_bytes).hexdigest()
require(gpu['inputsSha256'] == original_inputs_sha, 'Original inputs mismatch')
if args.kv_dtype == 'f32' or 'inputsSha256' in reference:
    require(reference.get('inputsSha256') == original_inputs_sha, 'Source original inputs mismatch')
original_inputs = {row['id']: row['tokens'] for row in json.loads(original_bytes)['sequences']}
require(reference.get('kvStorageDtype', 'f32') == args.kv_dtype, 'Reference KV lane mismatch')
decode_reference, decode_bytes, decode_input_bytes = None, None, None
require(bool(args.decode_reference) == bool(args.decode_inputs), 'Supply both decode reference and inputs')
if args.decode_reference:
    decode_path, decode_inputs_path = args.decode_reference, args.decode_inputs
    decode_bytes, decode_input_bytes = decode_path.read_bytes(), decode_inputs_path.read_bytes()
    decode_reference = json.loads(decode_bytes)
    require(decode_reference['artifactManifestSha256'] == gpu['manifestSha256'], 'Decode artifact mismatch')
    require(decode_reference.get('kvStorageDtype', 'f32') == args.kv_dtype, 'Decode KV lane mismatch')
    require(decode_reference['inputsSha256'] == hashlib.sha256(decode_input_bytes).hexdigest(), 'Decode inputs mismatch')
    decode_inputs = {row['id']: row['tokens'] for row in json.loads(decode_input_bytes)['sequences']}
protocol = None
if args.protocol:
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    require(protocol['schema'] == 'doppler.flash-next-learned-decode-protocol/v1', 'Unsupported decode protocol')
    for key, expected in [('inputsSha256', gpu['inputsSha256']), ('manifestSha256', gpu['manifestSha256']),
                          ('originalReferenceSha256', hashlib.sha256(reference_bytes).hexdigest()),
                          ('decodeReferenceSha256', hashlib.sha256(decode_bytes or b'').hexdigest()),
                          ('policySha256', hashlib.sha256(policy_bytes).hexdigest()), ('runSha256', gpu['runSha256'])]:
        require(protocol[key] == expected, 'Protocol identity mismatch: ' + key)
    require(protocol['policyId'] == policy_id and protocol['tolerancePolicy'] == policy, 'Policy changed')
    require(not protocol.get('acceptanceChanged'), 'Decode protocol changes acceptance')
    require(protocol['sourceGreedyChainVerifiedBeforeGpu'], 'Source greedy chain was not frozen before GPU use')
    require([row['id'] for row in gpu['rows']] == [row['id'] for row in protocol['chains']], 'Missing or reordered prompt cases')
    require(hashlib.sha256(decode_input_bytes).hexdigest() == protocol['decodeInputsSha256'], 'Protocol decode inputs mismatch')
if args.kv_dtype == 'f32':
    require(protocol is not None and decode_reference is not None, 'F32 decode comparison requires its explicit protocol')

def compare(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape and actual.size, 'Logit shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Non-finite comparison values')
    error = np.abs(actual - expected)
    limit = policy['absolute'] + policy['relative'] * np.abs(expected)
    return {'count': actual.size, 'maxAbsoluteError': float(error.max()),
            'rmsError': float(np.sqrt(np.mean(error * error))),
            'outsidePolicy': int(np.sum(error > limit)), 'passed': bool(np.all(error <= limit))}

def full_logits(file, digest, expected, directory, position):
    raw = Path(file).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'GPU tensor SHA-256 mismatch')
    saved = directory / expected['logitsFile']
    source_raw = saved.read_bytes()
    require(hashlib.sha256(source_raw).hexdigest() == expected['logitsSha256'], 'Source tensor SHA-256 mismatch')
    values = np.load(io.BytesIO(source_raw), allow_pickle=False)
    require(values.ndim == 2 and values.shape[0] == len(expected['samplePositions']) and values.dtype == np.float32, 'Invalid F32 source tensor')
    expected_logits = values[expected['samplePositions'].index(position)]
    actual = np.frombuffer(raw, dtype='<f4')
    return dict(compare(actual, expected_logits), actualArgmax=int(actual.argmax()), referenceArgmax=int(expected_logits.argmax()))

rows = []
for row in gpu['rows']:
    expected = reference['sequences'][row['id']]
    require(row['inputTokenCount'] == expected['tokens'], 'Prompt token count mismatch')
    require((row.get('kvStorageDtype') if protocol else row.get('kvStorageDtype', args.kv_dtype)) == args.kv_dtype, 'GPU KV lane mismatch')
    position = expected['tokens'] - 1
    result = {'id': row['id'], 'tokenIds': row['tokenIds'],
              'firstTokenAgrees': row['tokenId'] == expected['argmax'][position], 'layers': []}
    for layer in row.get('layers', []):
        require([r['position'] for r in layer['rows']] == expected['samplePositions'], 'Layer sample positions differ')
        values = [r['observed'] for r in layer['rows']]
        result['layers'].append({'layer': layer['layer'], 'scope': 'sampled stream coordinates',
                                 **compare(values, expected['layers'][str(layer['layer'])]['samples'])})
    if row.get('prefillLogitsFile'):
        result['logits'] = full_logits(row['prefillLogitsFile'], row['prefillLogitsSha256'], expected, reference_path.parent, position)
    result['passed'] = result['firstTokenAgrees'] and bool(result['layers'] or 'logits' in result) and all(l['passed'] for l in result['layers'])
    if 'logits' in result:
        result['passed'] = result['passed'] and result['logits']['passed']
        if protocol:
            result['passed'] = result['passed'] and result['logits']['actualArgmax'] == row['tokenId']
    result['fullLogitsChecked'] = 'logits' in result
    if decode_reference:
        result['decodeRows'] = []
        frames = row.get('stepLogits')
        if frames is None:
            raw_frames = row.get('logitFrames', [])
            require([f['step'] for f in raw_frames] == list(range(len(row['tokenIds']))), 'Missing or duplicated logit steps')
            frames = [dict(f, inputTokenCount=row['inputTokenCount'] + f['step'], tokenId=row['tokenIds'][f['step']]) for f in raw_frames]
        require(len(frames) == len(row['tokenIds']) and [f['step'] for f in frames] == list(range(len(frames))), 'Generation frame count mismatch')
        require(frames[0]['sha256'] == row['prefillLogitsSha256'], 'Prefill/frame binding mismatch')
        if protocol:
            require(len(frames) == protocol['maximumGenerationRowsPerPrompt'], 'Incomplete generation capture')
            chain = next(c for c in protocol['chains'] if c['id'] == row['id'])
            result['trajectoryAgrees'] = row['tokenIds'] == chain['expectedTokenIds']
        for observed in frames[1:]:
            step = observed['step']
            key = row['id'] + ('-next' if step == 1 else f'-next{step}')
            expected = decode_reference['sequences'][key]
            context = original_inputs[row['id']] + row['tokenIds'][:step]
            if context != decode_inputs[key]:
                result['decodeRows'].append(dict(step=step, passed=False, comparable=False, reason='Generated prefix differs from independent source prefix'))
                continue
            require(len(context) == expected['tokens'] == observed['inputTokenCount'], 'Decode position mismatch')
            comparison = full_logits(observed['file'], observed['sha256'], expected, decode_path.parent, len(context) - 1)
            agrees = observed['tokenId'] == expected['argmax'][-1]
            comparison.update(step=step, tokenId=observed['tokenId'], tokenAgrees=agrees)
            comparison['passed'] = comparison['passed'] and agrees and (protocol is None or comparison['actualArgmax'] == observed['tokenId'])
            result['decodeRows'].append(comparison)
        result['passed'] = result['passed'] and all(d['passed'] for d in result['decodeRows']) and result.get('trajectoryAgrees', True)
    rows.append(result)

result = {'schema': 'doppler.qwen38-learned-comparison/v1',
          'scope': 'Named unpadded Node observations only; sampled layers and retained last-position logits, no general model or performance qualification',
          'gpuSha256': hashlib.sha256(gpu_bytes).hexdigest(),
          'comparatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'kvStorageDtype': args.kv_dtype, 'normalModelQualified': False, 'acceptanceChanged': False,
          'referenceSha256': hashlib.sha256(reference_bytes).hexdigest(),
          'policySha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
          'policyId': policy_id, 'policy': policy, 'rows': rows,
          'passed': bool(rows) and all(row['passed'] for row in rows)}
if decode_reference:
    result['decodeReferenceSha256'] = hashlib.sha256(decode_bytes).hexdigest()
    result['decodeInputsSha256'] = hashlib.sha256(decode_input_bytes).hexdigest()
if protocol:
    result['protocolSha256'] = hashlib.sha256(protocol_bytes).hexdigest()
    result['decodeReferenceMethod'] = 'Independent cache-free full-prefix source recomputation'
    result['decodeRowsRequested'] = sum(len(row['decodeRows']) for row in rows)
    result['independentDecodeRowsChecked'] = sum(1 for row in rows for frame in row['decodeRows'] if 'count' in frame)
with output_path.open('x') as handle:
    handle.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({'passed': result['passed'], 'rows': len(rows), 'output': str(output_path)}))
if not result['passed']:
    raise SystemExit(1)
