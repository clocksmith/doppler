#!/usr/bin/env python3
"""Compare shared-prefix logit observations without using free-running token parity as a gate."""
import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path


def read_npy(path):
    data = path.read_bytes()
    assert data[:8] == b'\x93NUMPY\x01\x00', 'Expected NPY v1'
    start = 10 + int.from_bytes(data[8:10], 'little')
    header = ast.literal_eval(data[10:start].decode('ascii').strip())
    assert header['descr'] in ('<f4', '<f8') and header['fortran_order'] is False
    shape = header['shape']
    assert len(shape) == 2 and all(n > 0 for n in shape)
    kind = 'f' if header['descr'] == '<f4' else 'd'
    values = struct.unpack('<' + str(math.prod(shape)) + kind, data[start:])
    return [values[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]


def compare(reference, observed, tolerance):
    assert reference and len(reference) == len(observed)
    assert math.isfinite(tolerance) and tolerance >= 0
    assert all(math.isfinite(x) for x in (*reference, *observed))
    errors = [abs(a - b) for a, b in zip(reference, observed)]
    top = sorted(range(len(reference)), key=lambda k: reference[k], reverse=True)[:5]
    observed_top = max(range(len(observed)), key=lambda k: observed[k])
    denominator = math.sqrt(sum(x*x for x in reference) * sum(x*x for x in observed))
    return dict(maxAbsoluteError=max(errors), rmsError=math.sqrt(sum(x*x for x in errors) / len(errors)),
                cosineSimilarity=sum(a*b for a, b in zip(reference, observed)) / denominator if denominator else None,
                referenceTopToken=top[0], dopplerTopToken=observed_top,
                referenceMargin=reference[top[0]] - reference[top[1]],
                topReferenceTokens=[dict(token=k, reference=reference[k], doppler=observed[k], error=errors[k]) for k in top],
                passed=max(errors) <= tolerance)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(reference_root, candidate_root, contract_path, out):
    ref_path, obs_path = reference_root / 'reference.json', candidate_root / 'observation.json'
    reference, observed = json.loads(ref_path.read_text()), json.loads(obs_path.read_text())
    contract = json.loads(contract_path.read_text())
    assert reference['manifestSha256'] == observed['manifestSha256']
    assert digest(contract_path) == reference['contractSha256']
    assert observed.get('executionCompleted') and not observed.get('error') and not observed.get('cleanupError')
    assert len(reference['cases']) == 1 and len(contract['cases']) == 1
    case = reference['cases'][0]
    assert case['positions'] == contract['cases'][0]['positions'] == [r['position'] for r in observed['rows']]
    reference_file = reference_root / case['logits']['path']
    assert digest(reference_file) == case['logits']['sha256']
    expected = read_npy(reference_file)
    assert len(expected) == len(observed['rows'])
    tolerance = contract['acceptance']['implementationMaxAbsoluteLogitError']
    rows = []
    for values, row in zip(expected, observed['rows']):
        filename = candidate_root / row['file']
        assert digest(filename) == row['sha256']
        actual = struct.unpack('<' + str(row['count']) + 'f', filename.read_bytes())
        rows.append(dict(position=row['position'], **compare(values, actual, tolerance)))
    result = dict(schema='doppler.qwen-quantized-forced-logit-comparison/v1',
                  manifestSha256=reference['manifestSha256'], referenceSha256=digest(ref_path),
                  candidateSha256=digest(obs_path), contractSha256=digest(contract_path), toolSha256=digest(Path(__file__)),
                  maxAbsoluteLogitError=tolerance, rows=rows, passed=all(r['passed'] for r in rows),
                  scope='One fixed shared input trajectory; numerical diagnostic, not model quality or release qualification')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != 'topReferenceTokens'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference', 'candidate', 'contract', 'out']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.reference, args.candidate, args.contract, args.out)
