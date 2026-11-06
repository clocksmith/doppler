"""CPU-only F32 router replay at different batch geometries; no acceptance change.

Usage: python3 -P router-geometry-control.py <artifact> <prefix-control> <layer> <output.json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

artifact, prefix = map(Path, sys.argv[1:3])
layer, output = int(sys.argv[3]), Path(sys.argv[4])
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest_bytes = (artifact / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
receipt_bytes = (prefix / 'receipt.json').read_bytes()
receipt = json.loads(receipt_bytes)
assert receipt['manifestSha256'] == sha(manifest_bytes)
assert receipt['torch'] == torch.__version__
fixture_path = prefix / f'router-{layer}.npz'
fixture_bytes = fixture_path.read_bytes()
assert sha(fixture_bytes) == receipt['files'][fixture_path.name]
fixture = np.load(fixture_path, allow_pickle=False)
inputs = torch.from_numpy(fixture['input'])
expected = torch.from_numpy(fixture['logits'])
assert inputs.dtype == expected.dtype == torch.float32 and inputs.shape[0] > 0
descriptor = manifest['tensors'][f'model.language_model.layers.{layer}.mlp.gate.weight']
assert descriptor['dtype'] == 'F16' and descriptor.get('layout', 'row') == 'row'
raw = bytearray()
for span in descriptor.get('spans') or [dict(shardIndex=descriptor['shard'], offset=descriptor['offset'], size=descriptor['size'])]:
    with (artifact / manifest['shards'][span['shardIndex']]['filename']).open('rb') as handle:
        handle.seek(span['offset'])
        chunk = handle.read(span['size'])
        assert len(chunk) == span['size']
        raw.extend(chunk)
assert len(raw) == descriptor['size']
weights = torch.from_numpy(np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(descriptor['shape']))
torch.set_num_threads(16)
assert torch.get_float32_matmul_precision() == 'highest'
k = manifest['moeConfig']['numExpertsPerToken']
select = lambda logits: torch.softmax(logits, dim=-1).topk(k, dim=-1).indices.sort(dim=-1).values
expected_indices = select(expected)
report = dict(schema='doppler.flash-next-router-geometry-control/v1',
              scope='CPU F32 projection batch-geometry control; no GPU work or full-model qualification',
              normalModelQualified=False, torch=torch.__version__, threads=torch.get_num_threads(),
              manifestSha256=sha(manifest_bytes), prefixReceiptSha256=sha(receipt_bytes),
              fixtureSha256=sha(fixture_bytes), weightsSha256=sha(raw), layer=layer, rows=[])
with torch.inference_mode():
    for size in [inputs.shape[0], 512, 256, 32, 1]:
        actual = torch.cat([F.linear(chunk, weights) for chunk in inputs.split(size)], dim=0)
        error = (actual.double() - expected.double()).abs()
        changed = torch.nonzero((select(actual) != expected_indices).any(dim=1)).flatten().tolist()
        row = dict(batchRows=size, maxAbsoluteLogitDifference=float(error.max()),
                   differingLogitCount=int((actual != expected).sum()), changedExpertRows=changed)
        if size == inputs.shape[0]:
            assert torch.equal(actual, expected), 'full-geometry replay must reproduce the frozen upstream logits exactly'
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
output.write_text(json.dumps(report, indent=2) + '\n')
