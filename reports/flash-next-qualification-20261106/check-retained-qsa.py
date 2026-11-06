"""Check retained learned QSA scores/selections from their exact GPU inputs on CPU.

Usage: python3 -P check-retained-qsa.py <observation.json.gz> <output.json>
An operation-derived F32 rounding bound is diagnostic, not a new model tolerance.
"""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

observation_path, output_path = map(Path, sys.argv[1:])
raw = gzip.decompress(observation_path.read_bytes())
observation = json.loads(raw)
assert observation['completed'] and not observation.get('routingControl')
assert not observation['gpuErrors'] and not observation['cleanupFailures']
sha = lambda b: hashlib.sha256(b).hexdigest()
u = 2 ** -24
gamma = lambda n: n * u / (1 - n * u)
report = dict(schema='doppler.learned-qsa-same-input-check/v1',
              scope='CPU readback analysis of exact learned GPU query and pooled keys; no full-model qualification',
              observationSha256=sha(raw), normalModelQualified=False, rows=[])
for sequence in observation['rows']:
    for layer in sequence.get('qsaObservations', []):
        pooled_bytes = Path(layer['pooledFile']).read_bytes()
        assert sha(pooled_bytes) == layer['pooledSha256']
        width, heads = layer['headDim'], layer['queryHeads']
        keys = np.frombuffer(pooled_bytes, dtype='<f4').reshape(-1, width).astype(np.float64)
        for row in layer['rows']:
            position = row['position']
            # The pinned contract compresses four ordered visible tokens per block.
            blocks = (position + 1) // 4
            query = np.asarray(row['query'], dtype=np.float64).reshape(heads, width)
            expected_heads = query @ keys[:blocks].T
            positive = np.maximum(expected_heads, 0)
            expected = positive.sum(axis=0) / math.sqrt(width)
            dot_error = gamma(width) * (np.abs(query) @ np.abs(keys[:blocks]).T)
            bound = (dot_error.sum(axis=0) + gamma(heads) * (positive.sum(axis=0) + dot_error.sum(axis=0))) / math.sqrt(width)
            bound += 3 * u * (np.abs(expected) + bound)
            bound *= 1 + 2 ** -28
            actual = np.asarray(row['scores'][:blocks], dtype=np.float64)
            assert np.isfinite(actual).all()
            error = np.abs(actual - expected)
            failures = int((error > bound).sum())
            selected = np.asarray(row['blocks'], dtype=np.int64)
            assert len(selected) == len(np.unique(selected)) == min(512, blocks)
            assert ((selected >= 0) & (selected < blocks)).all()
            omitted = np.setdiff1d(np.arange(blocks), selected)
            exact_gpu_topk = not len(omitted) or float(actual[omitted].max()) <= float(actual[selected].min())
            stable_selection = not len(omitted) or float((expected[omitted] - bound[omitted]).max()) <= float((expected[selected] + bound[selected]).min())
            reconstructed = sorted([int(b) * 4 + offset for b in selected for offset in range(4)] + list(range(blocks * 4, position + 1)))
            assert len(reconstructed) == row['count']
            raw_indices = np.asarray(reconstructed, dtype='<u4')
            digest_encoding = layer.get('selectionDigestEncoding', 'legacy-truncated-low-byte')
            if digest_encoding == 'raw-u32-bytes/v2':
                assert reconstructed == row['indices']
                assert sha(raw_indices.tobytes()) == row['selectedSha256']
            else:
                assert sha(raw_indices.astype(np.uint8).tobytes()) == row['selectedSha256']
            wrong = expected_heads.sum(axis=0) / math.sqrt(width)
            wrong_rejected = bool((np.abs(actual - wrong) > bound).any())
            report['rows'].append(dict(layer=layer['layer'], position=position,
                                       comparedScores=blocks, maxAbsoluteError=float(error.max()),
                                       outsideDerivedBound=failures,
                                       maxErrorOverBound=float(np.max(np.divide(error, bound, out=np.zeros_like(error), where=bound > 0))),
                                       selectionMatchesActualGpuScores=exact_gpu_topk,
                                       selectionCompatibleWithRoundingBound=stable_selection,
                                       selectionDigestEncoding=digest_encoding,
                                       reconstructedFullIndexSha256=sha(raw_indices.tobytes()),
                                       visibleTokensAndTailCountMatch=True,
                                       negativeControlWithoutReluRejected=wrong_rejected))
assert report['rows']
report['numericalChecksPassed'] = all(r['outsideDerivedBound'] == 0 and r['selectionMatchesActualGpuScores'] and r['selectionCompatibleWithRoundingBound'] and r['negativeControlWithoutReluRejected'] for r in report['rows'])
report['selectionReceiptMeetsContract'] = all(r['selectionDigestEncoding'] == 'raw-u32-bytes/v2' for r in report['rows'])
report['passed'] = report['numericalChecksPassed'] and report['selectionReceiptMeetsContract']
output_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(passed=report['passed'], rows=len(report['rows']),
                     maxErrorOverBound=max(r['maxErrorOverBound'] for r in report['rows']))))
if not report['passed']:
    raise SystemExit(1)
