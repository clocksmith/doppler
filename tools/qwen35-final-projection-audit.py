#!/usr/bin/env python3
"""Separate final projection rounding from inherited hidden-state error offline."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('reference', Path(__file__).with_name('qwen35-f64-reference.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def decompose(actual, operands, weight, independently_evolved):
    """The final projection is linear; its same-operand error is separable."""
    assert all(np.isfinite(value).all() for value in (actual, operands, weight, independently_evolved))
    precise = operands.astype(np.float64) @ weight.astype(np.float64).T
    local = actual.astype(np.float64) - precise
    inherited = precise - independently_evolved
    np.testing.assert_allclose(local + inherited, actual - independently_evolved, rtol=0, atol=1e-12)
    return precise, local, inherited


def run(args):
    paths = {name: Path(getattr(args, name)) for name in ('contract', 'capture', 'baseline', 'independent')}
    contract, capture, baseline, independent = (reference.read_json(paths[name]) for name in paths)
    assert independent['contractSha256'] == reference.digest(paths['contract'].read_bytes())
    assert capture['archiveSha256'] == baseline['archiveSha256']
    assert capture['manifestIdentity'] == baseline['manifestIdentity']
    assert capture['observation']['target']['index'] == 2 and capture['observation']['target']['step'] == 3
    assert len(baseline['results']) == 4
    assert [(row['index'], row['step']) for row in baseline['results']] == [(2, step) for step in range(4)]
    observed = np.array(capture['observation']['observedSteps'], dtype=np.float32)
    excluded = capture['observation']['samplingExcludedTokenIds']
    mask = np.ones(observed.shape[1], dtype=bool)
    mask[excluded] = False
    for i, row in enumerate(baseline['results']):
        expected = np.frombuffer(base64.b64decode(row['logits']), dtype='<f4')
        np.testing.assert_array_equal(observed[i, mask], expected[mask])
    rows = [row for row in capture['observation']['timeline'] if row['opId'] == 'final_norm.out']
    assert len(rows) == 4
    operands = np.frombuffer(base64.b64decode(rows[-1]['capture']['dataBase64']), dtype='<f4')
    assert operands.shape == (1024,)
    inputs = contract['inputs']
    manifest = reference.read_json(inputs['manifest']['path'])
    assert reference.digest(Path(inputs['manifest']['path']).read_bytes()) == inputs['manifest']['sha256']
    weights, identities = reference.load_weights(Path(inputs['manifest']['path']).parent, manifest,
                                                  reference.read_json(inputs['pieceIndex']['path']))
    indexes = np.array([18050, 202605])
    assert all(mask[index] for index in indexes)
    expected = np.frombuffer(base64.b64decode(independent['results'][3]['logitsF64']), dtype='<f8')[indexes]
    actual = observed[-1, indexes].astype(np.float64)
    precise, local, inherited = decompose(actual, operands, weights['lm_head.weight'][indexes], expected)
    result = {'schema': 'doppler.qwen35-final-projection-audit/v1', 'qualified': False,
              'scope': 'Two failing logits, actual final normalization operands; F64 diagnostic, no reference activation',
              'identities': {name: reference.digest(path.read_bytes()) for name, path in paths.items()},
              'toolSha256': reference.digest(Path(__file__).read_bytes()), 'shards': identities,
              'allFourUnmaskedObservedLogitsUnchanged': True, 'samplingExcludedTokenIds': excluded, 'results': []}
    for j, index in enumerate(indexes):
        result['results'].append({'index': int(index), 'gpu': float(actual[j]), 'sameOperandF64': float(precise[j]),
                                  'independentlyEvolvedF64': float(expected[j]), 'localProjectionError': float(local[j]),
                                  'inheritedHiddenStateEffect': float(inherited[j]), 'totalError': float(actual[j] - expected[j])})
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['results'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'capture', 'baseline', 'independent', 'out'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
