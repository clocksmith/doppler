#!/usr/bin/env python3
"""Full error distribution against the approved vectors; never changes a target."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    bins = [0, 1e-6, 1e-5, .0001, .00025, .0005, .001, .002, .004, float('inf')]
    counts, _ = np.histogram(values, bins)
    return {'values': int(values.size), 'maximum': float(values.max()),
        'rms': float(np.sqrt(np.mean(values ** 2))), 'overTolerance': int(np.sum(values > .001)),
        'percentiles': dict(zip(['50', '90', '95', '99', '99.9', '99.99', '100'],
            map(float, np.percentile(values, [50, 90, 95, 99, 99.9, 99.99, 100])))),
        'histogram': [{'lowerInclusive': low, 'upperExclusive': high if np.isfinite(high) else None,
            'count': int(count)} for low, high, count in zip(bins, bins[1:], counts)]}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture', required=True)
parser.add_argument('--binding', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
root = Path('artifacts/whole-model-f64-complete-20261011')
index_path = root / 'independent-output-index.json'
index, proposal, approval = [read(root / name) for name in
    ('independent-output-index.json', 'proposal.json', 'approval.json')]
assert sha('tools/qwen35-f64-reference.py') == proposal['toolSha256']
expected = {}
for shard in index['shards']:
    assert sha(shard['path']) == shard['sha256']
    record = read(shard['path'])
    assert record['toolSha256'] == proposal['toolSha256']
    assert record['contractSha256'] == proposal['contracts'][shard['index']]['sha256']
    for row in record['results']:
        key = row['index'], row['step']
        assert key not in expected
        expected[key] = np.frombuffer(base64.b64decode(row['logitsF64']), dtype='<f8')
capture, binding = read(args.capture), read(args.binding)
assert capture['executionCompleted'] and capture['physicalExecutionCompleted'] and capture['fullPrefixes']
assert not capture['prefixesOnly'] and capture['observation'] is None
assert capture['declaredShaderMismatches'] == []
assert capture['archiveSha256'] == binding['archiveSha256']
assert capture['manifestIdentity'] == 'sha256:' + binding['manifestSha256']
assert len(capture['results']) == 57 and len(expected) == 55
observed = {}
for row in capture['results']:
    key = row['index'], row['step']
    if key in observed:
        assert observed[key]['logits'] == row['logits'], 'Resident reuse changed output'
    observed[key] = row
assert observed.keys() == expected.keys()
rows, errors = [], []
for key, target in expected.items():
    row = observed[key]
    contract = read(proposal['contracts'][key[0]]['path'])
    assert row['inputIds'] == contract['inputIds'][:contract['prefixLengths'][0]]
    actual = np.frombuffer(base64.b64decode(row['logits']), dtype='<f4').astype(np.float64)
    assert actual.shape == target.shape and np.isfinite(actual).all() and np.isfinite(target).all()
    error = np.abs(actual - target)
    errors.append(error)
    rows.append({'index': key[0], 'step': key[1], 'worstVocabulary': int(np.argmax(error)),
                 **distribution(error)})
result = {'schema': 'doppler.approved-f64-error-distribution/v1', 'qualified': False,
    'tolerance': .001, 'captureSha256': sha(args.capture), 'archiveSha256': capture['archiveSha256'],
    'manifestIdentity': capture['manifestIdentity'], 'approvalSha256': sha(root / 'approval.json'),
    'approvedIndexSha256': sha(index_path), 'scriptSha256': sha(__file__),
    'failedPrefixes': sum(row['maximum'] > .001 for row in rows), 'prefixes': rows,
    'completeVocabulary': distribution(np.concatenate(errors))}
Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'failedPrefixes': result['failedPrefixes'], **result['completeVocabulary']}, indent=2))
