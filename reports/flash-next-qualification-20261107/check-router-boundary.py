"""Exact-dot diagnostic at the first changed expert boundary; no GPU feedback.

Usage: python3 check-router-boundary.py GPU ROUTING_COMPARISON CPU_TRACE ARTIFACT OUT
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flash-next-qualification-20261106'))
from verified_tensor_io import read_tensor


def require(condition, message):
    if not condition:
        raise ValueError(message)


gpu_path, comparison_path, prefix, artifact, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
gpu_bytes, comparison_bytes, manifest_bytes, parent_bytes = [p.read_bytes() for p in [gpu_path, comparison_path, artifact/'manifest.json', prefix/'receipt.json']]
gpu, comparison, manifest, parent = map(json.loads, [gpu_bytes, comparison_bytes, manifest_bytes, parent_bytes])
require(gpu['completed'] and parent['completed'] and comparison['gpuSha256'] == sha(gpu_bytes), 'Incomplete or unmatched observations')
require(comparison['sourceTraceSha256'] == sha(parent_bytes), 'Different source trace')
require(gpu['manifestSha256'] == parent['manifestSha256'] == sha(manifest_bytes), 'Different weights')
layer = comparison['firstChangedLayer']
require(layer is not None, 'No changed expert set')
record = next(r for r in gpu['rows'][0]['routerObservations'] if r['layer'] == layer)
gpu_input, gpu_logits = [read_tensor(record[k]) for k in ['input', 'logits']]
filename = f'router-{layer}.npz'
require(sha((prefix/filename).read_bytes()) == parent['files'][filename], 'Source trace hash mismatch')
source = np.load(prefix/filename, allow_pickle=False)
descriptor = manifest['tensors'][f'model.language_model.layers.{layer}.mlp.gate.weight']
require(descriptor['dtype'] == 'F16', 'Expected exact F16-stored routing weights')
raw = bytearray()
for span in descriptor.get('spans') or [dict(shardIndex=descriptor['shard'], offset=descriptor['offset'], size=descriptor['size'])]:
    with (artifact/manifest['shards'][span['shardIndex']]['filename']).open('rb') as handle:
        handle.seek(span['offset'])
        data = handle.read(span['size'])
        require(len(data) == span['size'], 'Incomplete router weights')
        raw.extend(data)
weights = np.frombuffer(raw, '<f2').astype('f8').reshape(descriptor['shape'])
position = comparison['rows'][layer]['first']['position']
first = comparison['rows'][layer]['first']
changed = sorted(set(first['sourceExperts']) ^ set(first['gpuExperts']))
rows = []
for expert in changed:
    exact = lambda values: math.fsum(float(a)*float(b) for a,b in zip(values[position],weights[expert]))
    rows.append(dict(expert=expert, sourceStored=float(source['logits'][position,expert]), gpuStored=float(gpu_logits[position,expert]),
                     sourceInputExactDot=exact(source['input']), gpuInputExactDot=exact(gpu_input)))
report = dict(scope='CPU-only exact-dot diagnostic, no expert replacement or acceptance change', layer=layer, position=position,
              gpuSha256=sha(gpu_bytes), comparisonSha256=sha(comparison_bytes), parentSha256=sha(parent_bytes), manifestSha256=sha(manifest_bytes),
              scriptSha256=sha(Path(__file__).read_bytes()), weightSha256=sha(raw), rows=rows,
              inputMaxAbsolute=float(np.abs(gpu_input[position].astype('f8')-source['input'][position].astype('f8')).max()))
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
