"""Compare observed stages with independent CPU traces for layers 0..N.

Usage: python3 -P compare-upstream-stages.py <gpu.json> <out.json> [<first-layer>] <cpu-layer.json> ...
No model acceptance thresholds are applied or changed by this diagnostic.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

gpu_path, output = map(Path, sys.argv[1:3])
first = int(sys.argv[3]) if sys.argv[3].isdigit() else 0
source_paths = list(map(Path, sys.argv[4:] if sys.argv[3].isdigit() else sys.argv[3:]))
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes = gpu_path.read_bytes()
gpu = json.loads(gpu_bytes)
assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
report = dict(scope='Observed GPU versus independent CPU stage slices; not full-tensor or model qualification',
              normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes),
              scriptSha256=sha(Path(__file__).read_bytes()), sourceTraces=[], rows=[])
for layer, source_path in enumerate(source_paths, start=first):
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    assert source['diagnosticCompleted'] and source['manifestSha256'] == gpu['manifestSha256']
    assert source['inputsSha256'] == gpu['inputsSha256']
    report['sourceTraces'].append(dict(layer=layer, file=str(source_path), sha256=sha(source_bytes)))
    for observed in [r for r in gpu['tensors'] if r['layer'] == layer]:
        role, stage = observed['role'], observed['stage']
        if role == 'linear' and stage == 'attn.post_input_norm':
            role, stage = 'attn_hc', 'branchInput'
        expected = next((r for r in source['tensors'] if (r['role'], r['stage']) == (role, stage)), None)
        assert expected, f'No independent source observation for layer {layer}: {role}/{stage}'
        raw = Path(observed['file']).read_bytes()
        assert sha(raw) == observed['sha256']
        actual = np.frombuffer(raw, dtype='<f4').reshape(observed['shape'])
        reference_path = source_path.parent / expected['file']
        reference_bytes = reference_path.read_bytes()
        assert sha(reference_bytes) == expected['sha256']
        reference = np.load(reference_path, allow_pickle=False)
        positions = expected['positions']
        assert positions
        if observed.get('positions') is not None:
            assert observed['positions'] == positions
        else:
            actual = actual[positions]
        assert actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
        difference = actual.astype(np.float64) - reference
        report['rows'].append(dict(layer=layer, role=observed['role'], stage=observed['stage'], positions=positions,
            gpuSha256=observed['sha256'], sourceSha256=expected['sha256'],
            maxAbsoluteDifference=float(np.abs(difference).max()), rmsDifference=float(np.sqrt((difference ** 2).mean())),
            differingF32Values=int(np.count_nonzero(actual.view(np.uint32) != reference.view(np.uint32)))))
assert report['rows']
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps([r for r in report['rows'] if r['stage'] in ('input', 'norm', 'down', 'branchInput')], indent=2))
