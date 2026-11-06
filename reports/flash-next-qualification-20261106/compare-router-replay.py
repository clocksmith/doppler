"""Locate expert-set divergence using actual GPU top-k, never CPU tie choices.

Usage: python3 -P compare-router-replay.py <gpu.json> <replay.json> <source-prefix> <artifact> <out.json>
This reports diagnostics; it never replaces or relaxes the model comparator.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

gpu_path, replay_path, source, artifact, output = map(Path, sys.argv[1:])
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes, replay_bytes = gpu_path.read_bytes(), replay_path.read_bytes()
gpu, replay = json.loads(gpu_bytes), json.loads(replay_bytes)
manifest_bytes = (artifact / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
parent_bytes = (source / 'receipt.json').read_bytes()
parent = json.loads(parent_bytes)
assert gpu['completed'] and replay['completed']
assert not gpu.get('routingControl') and not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert not replay['gpuErrors'] and not replay['cleanupFailures']
assert replay['observationSha256'] == sha(gpu_bytes)
assert gpu['manifestSha256'] == parent['manifestSha256'] == sha(manifest_bytes)
assert gpu['inputsSha256'] == parent['inputsSha256']
report = dict(scope='Actual GPU top-k replay versus independent source expert choices; not model acceptance',
              normalModelQualified=False, observationSha256=sha(gpu_bytes), replaySha256=sha(replay_bytes),
              sourceReceiptSha256=sha(parent_bytes), manifestSha256=sha(manifest_bytes),
              scriptSha256=sha(Path(__file__).read_bytes()), rows=[])
for row in replay['rows']:
    layer = row['layer']
    name = f'router-{layer}.npz'
    if not (source / name).exists():
        report['rows'].append(dict(layer=layer, compared=False, reason='No retained independent source router fixture in this prefix'))
        continue
    raw = Path(row['file']).read_bytes()
    assert sha(raw) == row['sha256']
    actual = np.frombuffer(raw, dtype='<u4').reshape(row['shape'])
    assert sha((source / name).read_bytes()) == parent['files'][name]
    ref = np.load(source / name, allow_pickle=False)
    expected = ref['experts']
    changed = np.flatnonzero((np.sort(actual, axis=1) != np.sort(expected, axis=1)).any(axis=1)).tolist()
    result = dict(layer=layer, compared=True, changedPositions=changed,
                  sourceFixtureSha256=parent['files'][name], sourceLogitsSha256=row['sourceLogitsSha256'])
    if changed:
        position = changed[0]
        record = next(r for r in gpu['rows'][0]['routerObservations'] if r['layer'] == layer)
        assert record['logits']['sha256'] == row['sourceLogitsSha256']
        values = {}
        for key in ('input', 'logits'):
            blob = Path(record[key]['file']).read_bytes()
            assert sha(blob) == record[key]['sha256']
            values[key] = np.frombuffer(blob, dtype='<f4').reshape(record[key]['shape'])[position]
        descriptor = manifest['tensors'][f'model.language_model.layers.{layer}.mlp.gate.weight']
        assert descriptor['dtype'] == 'F16'
        weight_bytes = bytearray()
        for span in descriptor.get('spans') or [dict(shardIndex=descriptor['shard'], offset=descriptor['offset'], size=descriptor['size'])]:
            with (artifact / manifest['shards'][span['shardIndex']]['filename']).open('rb') as handle:
                handle.seek(span['offset'])
                blob = handle.read(span['size'])
                assert len(blob) == span['size']
                weight_bytes.extend(blob)
        weights = np.frombuffer(weight_bytes, dtype='<f2').astype(np.float64).reshape(descriptor['shape'])
        lost = sorted(set(expected[position].tolist()) - set(actual[position].tolist()))
        gained = sorted(set(actual[position].tolist()) - set(expected[position].tolist()))
        experts = sorted(set(lost + gained))
        exact = lambda x, expert: math.fsum(float(a) * float(b) for a, b in zip(x, weights[expert]))
        input_available = ref['input'].shape == (actual.shape[0], weights.shape[1])
        result['firstDifference'] = dict(position=position, lost=lost, gained=gained,
            sourceInputAvailable=input_available,
            inputMaxAbsoluteDifference=float(np.max(np.abs(values['input'].astype(np.float64) - ref['input'][position]))) if input_available else None,
            boundaryLogits=[dict(expert=e, sourceStored=float(ref['logits'][position, e]), gpuStored=float(values['logits'][e]),
                                sourceInputF64Dot=exact(ref['input'][position], e) if input_available else None,
                                gpuInputF64Dot=exact(values['input'], e)) for e in experts])
    report['rows'].append(result)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('scope', 'rows')}, indent=2))
