"""Exact dyadic router margins on captured inputs; no GPU or acceptance change.

Usage: assess-router-inputs.py LAYER25_PROTOCOL ARTIFACT ROUTING_TRACE
       GPU_INPUT_F32 GPU_LOGITS_F32 NEW_OUTPUT
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

protocol_path, artifact, cpu_root, input_path, logits_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
p = json.loads(protocol_path.read_bytes())
manifest_raw = (artifact / 'manifest.json').read_bytes()
assert sha(manifest_raw) == p['manifestSha256']
manifest = json.loads(manifest_raw)
name = 'mlp.gate.weight'
weight = manifest['tensors'][f'model.language_model.layers.25.{name}']
spans = weight.get('spans', [dict(shardIndex=weight.get('shard'), offset=weight.get('offset'), size=weight.get('size'))])
parts = []
for span in spans:
    with (artifact / manifest['shards'][span['shardIndex']]['filename']).open('rb') as handle:
        handle.seek(span['offset'])
        part = handle.read(span['size'])
        assert len(part) == span['size']
        parts.append(part)
raw = b''.join(parts)
assert sha(raw) == p['weights'][name]['sha256']
w = np.frombuffer(raw, '<f2').astype('f4').reshape(512, 2560)
input_raw, logits_raw = input_path.read_bytes(), logits_path.read_bytes()
x = np.frombuffer(input_raw, '<f4').reshape(2304, 2560)
logits = np.frombuffer(logits_raw, '<f4').reshape(2304, 512)
receipt_raw = (cpu_root / 'receipt.json').read_bytes()
assert sha(receipt_raw) == p['routerTraceReceiptSha256']
receipt = json.loads(receipt_raw)
assert sha((cpu_root / 'router-25.npz').read_bytes()) == receipt['files']['router-25.npz']
with np.load(cpu_root / 'router-25.npz', allow_pickle=False) as data:
    source_x, source_logits = data['input'].copy(), data['logits'].copy()
def exact_dot(values, weights):
    return sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b)) for a, b in zip(values, weights)), Fraction(0))
rows = []
for position, a, b in [(1717, 493, 330), (2271, 311, 411)]:
    source = exact_dot(source_x[position], w[a]) - exact_dot(source_x[position], w[b])
    gpu_input = exact_dot(x[position], w[a]) - exact_dot(x[position], w[b])
    rows.append(dict(position=position, expertA=a, expertB=b,
                     capturedSourceMargin=float(source_logits[position, a] - source_logits[position, b]),
                     capturedGpuMargin=float(logits[position, a] - logits[position, b]),
                     exactSourceInputMargin=float(source), exactGpuInputMargin=float(gpu_input),
                     exactSourceInputMarginDyadic=str(source), exactGpuInputMarginDyadic=str(gpu_input),
                     exactMarginsHaveSameSign=(source > 0) == (gpu_input > 0)))
result = dict(scope='Exact dyadic router margins on separately captured actual inputs; not model qualification',
              normalModelQualified=False, acceptanceChanged=False, manifestSha256=p['manifestSha256'],
              inputSha256=sha(input_raw), logitsSha256=sha(logits_raw), routerWeightsSha256=sha(raw),
              cpuRoutingTraceReceiptSha256=sha(receipt_raw),
              producerSha256=sha(Path(__file__).read_bytes()), rows=rows)
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(rows, indent=2))
