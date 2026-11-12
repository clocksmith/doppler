"""Score forced-routing GPU observations against the F32 CPU reference, unchanged gate.

Usage: python3 -P compare-forced-routing.py OUT.json REFERENCE.json POLICY.json INPUTS.json SEQUENCE LABEL=OBSERVATION.json...

Uses the same arithmetic as artifacts/qwen38-flash-next-20261013/compare-learned-reference.py
(abs <= absolute + relative*|ref| over every last-position logit). That comparator refuses
substituted executions by design; this script never marks a counterfactual as qualification.
"""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

if len(sys.argv) < 7:
    raise SystemExit(__doc__)
out, reference_path, policy_path, inputs_path = map(Path, sys.argv[1:5])
sequence = sys.argv[5]
observations = [arg.split('=', 1) for arg in sys.argv[6:]]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


reference_bytes, policy_bytes, inputs_bytes = reference_path.read_bytes(), policy_path.read_bytes(), inputs_path.read_bytes()
reference = json.loads(reference_bytes)
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = json.loads(policy_bytes)['tolerancePolicies'][policy_id]
require(policy['absolute'] == 0.05 and policy['relative'] == 0.05, 'Tolerance changed')
expected = reference['sequences'][sequence]
position = expected['tokens'] - 1
source_raw = (reference_path.parent / expected['logitsFile']).read_bytes()
require(sha(source_raw) == expected['logitsSha256'], 'Source tensor SHA-256 mismatch')
values = np.load(io.BytesIO(source_raw), allow_pickle=False)
reference_logits = values[expected['samplePositions'].index(position)].astype(np.float64)

rows = []
for label, path in observations:
    raw_observation = Path(path).read_bytes()
    gpu = json.loads(raw_observation)
    require(gpu.get('completed') and not gpu.get('failure'), f'{label}: GPU observation did not complete')
    require(not gpu['gpuErrors'] and not gpu['cleanupFailures'], f'{label}: GPU or cleanup errors remain')
    require(gpu['referenceSha256'] == sha(reference_bytes), f'{label}: reference binding')
    require(gpu['inputsSha256'] == sha(inputs_bytes), f'{label}: inputs binding')
    require(gpu['manifestSha256'] == reference['artifactManifestSha256'], f'{label}: artifact binding')
    row = next(r for r in gpu['rows'] if r['id'] == sequence)
    require(row['inputTokenCount'] == expected['tokens'] and row.get('kvStorageDtype', 'f32') == 'f32', f'{label}: lane')
    raw = Path(row['prefillLogitsFile']).read_bytes()
    require(sha(raw) == row['prefillLogitsSha256'], f'{label}: GPU tensor SHA-256 mismatch')
    actual = np.frombuffer(raw, dtype='<f4').astype(np.float64)
    require(actual.shape == reference_logits.shape and np.isfinite(actual).all(), f'{label}: logits')
    error = np.abs(actual - reference_logits)
    limit = policy['absolute'] + policy['relative'] * np.abs(reference_logits)
    control = gpu.get('counterfactual') or {}
    rows.append({
        'label': label, 'observationSha256': sha(raw_observation), 'prefillLogitsSha256': row['prefillLogitsSha256'],
        'routingControlMode': control.get('mode'), 'count': int(actual.size),
        'maxAbsoluteError': float(error.max()), 'argmaxAbsoluteError': int(error.argmax()),
        'rmsError': float(np.sqrt(np.mean(error * error))),
        'outsidePolicy': int(np.sum(error > limit)), 'passed': bool(np.all(error <= limit)),
        'greedyToken': row['tokenId'], 'actualArgmax': int(actual.argmax()),
        'referenceArgmax': int(reference_logits.argmax()), 'greedyAgrees': row['tokenId'] == expected['argmax'][position],
    })
result = dict(schema='doppler.flash-next-forced-routing-comparison/v1', sequenceId=sequence, policyId=policy_id,
              policy=policy, policySha256=sha(policy_bytes), referenceSha256=sha(reference_bytes),
              inputsSha256=sha(inputs_bytes), comparatorSha256=sha(Path(__file__).read_bytes()),
              acceptanceChanged=False, normalModelQualified=False, rows=rows)
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([{k: r[k] for k in ['label', 'maxAbsoluteError', 'outsidePolicy', 'greedyToken', 'passed']} for r in rows]))
