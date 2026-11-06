"""Compare one independently observed sparse query and its selected token set.

Usage: python3 -P compare-sparse-query.py <gpu-trace.json> <cpu-prefix> <layer> <position> <out.json>
The independent prefix remains authoritative; this is not an acceptance gate.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

gpu_path, prefix = map(Path, sys.argv[1:3])
layer_id, position = map(int, sys.argv[3:5])
output = Path(sys.argv[5])
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes, receipt_bytes = gpu_path.read_bytes(), (prefix / 'receipt.json').read_bytes()
gpu, receipt = json.loads(gpu_bytes), json.loads(receipt_bytes)
assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['manifestSha256'] == receipt['manifestSha256']
assert gpu['inputsSha256'] == receipt['inputsSha256']
name = f'qsa-{layer_id}-{position}.npz'
blob = (prefix / name).read_bytes()
assert sha(blob) == receipt['files'][name]
source = np.load(prefix / name, allow_pickle=False)
layer = next(r for r in gpu['qsaObservations'] if r['layer'] == layer_id)
row = next(r for r in layer['rows'] if r['position'] == position)
actual, expected = set(row['indices']), set(source['selected'].tolist())
assert layer['selectionDigestEncoding'] == 'raw-u32-bytes/v2'
assert sha(np.asarray(row['indices'], dtype='<u4').tobytes()) == row['selectedSha256']
scores = source['scores'].reshape(-1)
order = np.sort(scores)[::-1]
assert len(order) > 512
report = dict(scope='Independent source sparse query versus GPU readback; not model acceptance', normalModelQualified=False,
              sourceReceiptSha256=sha(receipt_bytes), gpuTraceSha256=sha(gpu_bytes), sourceQsaSha256=sha(blob),
              scriptSha256=sha(Path(__file__).read_bytes()), layer=layer_id, position=position,
              sourceShapes={k: list(source[k].shape) for k in source.files},
              addedTokens=sorted(actual - expected), removedTokens=sorted(expected - actual),
              sourceCutoffGap=float(order[511] - order[512]), sourceCutoffScore=float(order[511]),
              queryMaxAbsoluteDifference=float(np.abs(np.asarray(row['query']).reshape(source['query'].shape) - source['query']).max()),
              scoreMaxAbsoluteDifference=float(np.abs(np.asarray(row['scores'][:len(scores)]) - scores).max()))
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
