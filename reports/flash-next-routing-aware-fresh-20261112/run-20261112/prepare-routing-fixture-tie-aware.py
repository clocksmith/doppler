"""Convert the retained CPU source router observations into the forced-routing fixture.

Usage: python3 -P prepare-routing-fixture.py SOURCE_TRACE OUT MANIFEST INPUTS REFERENCE SEQUENCE

SOURCE_TRACE is the F32 source observer output (router-<L>.npz + receipt.json) from
reports/flash-next-new-prompts-20261110. CPU only; never reads GPU output.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if len(sys.argv) != 7:
    raise SystemExit(__doc__)
source, out, manifest, inputs, reference = map(Path, sys.argv[1:6])
sequence = sys.argv[6]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


receipt_bytes = (source / 'receipt.json').read_bytes()
receipt = json.loads(receipt_bytes)
require(receipt.get('completed') and receipt.get('samplesAndRouterMetadataMatchReference'), 'Incomplete source trace')
require(receipt.get('sequenceId', sequence) == sequence, 'Different sequence')
identity = dict(manifestSha256=sha(manifest.read_bytes()), inputsSha256=sha(inputs.read_bytes()),
                referenceSha256=sha(reference.read_bytes()))
for key, value in identity.items():
    require(receipt[key] == value, f'Source trace {key} mismatch')
tokens = next(s['tokens'] for s in json.loads(inputs.read_bytes())['sequences'] if s['id'] == sequence)
out.mkdir()
files, margins = [], []
for layer in range(48):
    name = f'router-{layer}.npz'
    raw = (source / name).read_bytes()
    require(sha(raw) == receipt['files'][name], f'{name} hash')
    data = np.load(source / name)
    logits, experts, scores = data['logits'], data['experts'], data['scores']
    T, E = logits.shape
    K = experts.shape[1]
    require(T == len(tokens) and E == 512 and K == 10, f'{name} geometry')
    require(logits.dtype == np.float32 and scores.dtype == np.float32, f'{name} dtype')
    require(np.isfinite(logits).all() and np.isfinite(scores).all(), f'{name} finite')
    require((np.sort(experts, axis=1)[:, 1:] != np.sort(experts, axis=1)[:, :-1]).all(), f'{name} duplicate experts')
    # Router consistency: the retained experts are a top-k of the retained logits. An exact
    # rank-k/k+1 tie admits more than one top-k set, so compare values, not one argsort's indices.
    order = np.argsort(-logits.astype(np.float64), axis=1, kind='stable')
    selected = np.zeros(logits.shape, dtype=bool)
    np.put_along_axis(selected, experts.astype(np.int64), True, axis=1)
    require((np.where(selected, logits, np.inf).min(axis=1) >= np.where(selected, -np.inf, logits).max(axis=1)).all(),
            f'{name} experts are not top-k of logits')
    require(np.allclose(scores.sum(axis=1), 1.0, atol=1e-5), f'{name} weights are not normalized')
    ordered = np.take_along_axis(logits, order[:, :K + 1], axis=1)
    margin = ordered[:, K - 1] - ordered[:, K]
    margins.append(margin)
    item = {'layer': layer}
    for key, array, suffix in [('experts', experts.astype('<u4'), 'u32'), ('scores', scores.astype('<f4'), 'f32'),
                               ('logits', logits.astype('<f4'), 'f32')]:
        blob = np.ascontiguousarray(array).tobytes()
        path = out / f'{key}-{layer}.{suffix}'
        path.write_bytes(blob)
        item[key] = {'file': path.name, 'shape': list(array.shape), 'sha256': sha(blob)}
    files.append(item)
margins = np.stack(margins)
fixture = dict(schema='doppler.flash-next-routing-selection-fixture/v1', sequenceId=sequence,
               scope='CPU F32 source router selections (independent reference producer); never GPU output',
               layers=48, numTokens=len(tokens), numExperts=512, topK=10, **identity,
               sourceTraceReceiptSha256=sha(receipt_bytes), producerSha256=receipt['producerSha256'],
               converterSha256=sha(Path(__file__).read_bytes()),
               rank10Margin={'min': float(margins.min()),
                             'tokenLayersBelow': {str(t): int((margins < t).sum()) for t in [1e-5, 1e-4, 1e-3]},
                             'tokenLayers': int(margins.size)},
               files=files)
(out / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
print(json.dumps({'completed': True, 'rank10Margin': fixture['rank10Margin']}))
