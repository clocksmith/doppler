#!/usr/bin/env python3
"""Compare retained independent vectors after computation; never activate them."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--binding', default='artifacts/whole-model-f64-complete-20261011/repaired-replay-binding.json')
parser.add_argument('--label', default='repairedDoppler')
args = parser.parse_args()
root = Path(__file__).parent
spec = importlib.util.spec_from_file_location('independent_f64', 'tools/qwen35-f64-reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
proposal = read(root / 'proposal.json')
assert sha(spec.origin) == proposal['toolSha256']
index = read(root / 'independent-output-index.json')
rows = []
for entry in index['shards']:
    path = Path(entry['path'])
    assert sha(path) == entry['sha256']
    report = read(path)
    assert report['toolSha256'] == proposal['toolSha256']
    contract = proposal['contracts'][entry['index']]
    assert report['contractSha256'] == contract['sha256'] == sha(contract['path'])
    rows.extend(report['results'])
assert len(rows) == 55 and len({(r['index'], r['step']) for r in rows}) == 55
capture = read(args.capture)
binding = read(args.binding)
assert capture['executionCompleted'] and capture['physicalExecutionCompleted']
assert capture['fullPrefixes'] and not capture['prefixesOnly']
assert capture['observation'] is None and capture['declaredShaderMismatches'] == []
assert capture['archiveSha256'] == binding['archiveSha256']
assert capture['manifestIdentity'] == 'sha256:' + binding['manifestSha256']
assert len(capture['results']) == 57
observed = {}
for row in capture['results']:
    key = row['index'], row['step']
    if key in observed:
        assert row['logits'] == observed[key]['logits'], 'Reuse changed output'
    observed[key] = row
comparisons = []
inputs = read(proposal['contracts'][0]['path'])['inputs']
historical = read(inputs['historicalReference']['path'])
source = read(inputs['sourceChunk']['path'])
source_rows = {(row['index'], row['step']): row for row in source['results']}
for name in ('historicalReference', 'sourceChunk'):
    assert sha(inputs[name]['path']) == inputs[name]['sha256']
for row in rows:
    contract = read(proposal['contracts'][row['index']]['path'])
    target = observed[row['index'], row['step']]
    assert target['inputIds'] == contract['inputIds'][:contract['prefixLengths'][0]]
    values = np.frombuffer(base64.b64decode(row['logitsF64']), '<f8')
    comparisons.append({key: value for key, value in row.items() if key != 'logitsF64'})
    comparisons[-1]['comparisons'][args.label] = reference.comparison(values, target['logits'], .001)
    repaired = np.frombuffer(base64.b64decode(target['logits']), '<f4')
    selected = {row['greedyToken'], comparisons[-1]['comparisons'][args.label]['argmaxDifference']}
    comparisons[-1]['selectedValues'] = [{'tokenId': int(token), 'independentF64': float(values[token]),
                                          args.label: float(repaired[token])} for token in sorted(selected)]
    targets = {'historical': historical['expected'][row['index']]['steps'][row['step']]['logits'],
               'sourceF32': source_rows[row['index'], row['step']]['logits']}
    comparisons[-1]['otherTargets'] = {
        name: reference.comparison(np.frombuffer(base64.b64decode(encoded), '<f4').astype(np.float64),
                                   target['logits'], .001) for name, encoded in targets.items()}
summary = {}
for name in comparisons[0]['comparisons']:
    values = [row['comparisons'][name] for row in comparisons]
    summary[name] = {'prefixes': len(values), 'failedPrefixes': sum(not value['pass'] for value in values),
                     'maximumAbsoluteDifference': max(value['maxAbsDifference'] for value in values),
                     'greedyTokenMatches': sum(value['greedyToken'] == row['greedyToken']
                                               for value, row in zip(values, comparisons))}
other_targets = {}
for name in ('historical', 'sourceF32'):
    values = [row['otherTargets'][name] for row in comparisons]
    other_targets[name] = {'prefixes': 55, 'failedPrefixes': sum(not value['pass'] for value in values),
                           'maximumAbsoluteDifference': max(value['maxAbsDifference'] for value in values)}
result = {'schema': 'doppler.complete-f64-comparison/v1', 'referenceActivated': False,
          'qualified': False, 'tolerance': .001, 'captureSha256': sha(args.capture),
          'archiveSha256': capture['archiveSha256'], 'manifestIdentity': capture['manifestIdentity'],
          'comparatorSha256': sha(__file__), 'summary': summary, 'otherTargets': other_targets,
          'results': comparisons}
Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print(json.dumps(other_targets, indent=2))
