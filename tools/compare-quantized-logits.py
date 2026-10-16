#!/usr/bin/env python3
"""Compare shared-prefix logit observations without using free-running token parity as a gate."""
import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path


def read_tensor(path):
    data = path.read_bytes()
    assert data[:8] == b'\x93NUMPY\x01\x00', 'Expected NPY v1'
    start = 10 + int.from_bytes(data[8:10], 'little')
    header = ast.literal_eval(data[10:start].decode('ascii').strip())
    assert header['descr'] in ('<f4', '<f8') and header['fortran_order'] is False
    shape = header['shape']
    assert shape and all(type(n) is int and n > 0 for n in shape)
    kind = 'f' if header['descr'] == '<f4' else 'd'
    values = struct.unpack('<' + str(math.prod(shape)) + kind, data[start:])
    return shape, values


def read_npy(path):
    shape, values = read_tensor(path)
    assert len(shape) == 2
    return [values[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]


def compare(reference, observed, tolerance, rms_tolerance=None):
    assert len(reference) >= 2 and len(reference) == len(observed)
    assert math.isfinite(tolerance) and tolerance >= 0
    assert rms_tolerance is None or (math.isfinite(rms_tolerance) and rms_tolerance >= 0)
    assert all(math.isfinite(x) for x in (*reference, *observed))
    errors = [abs(a - b) for a, b in zip(reference, observed)]
    top = sorted(range(len(reference)), key=lambda k: reference[k], reverse=True)[:5]
    observed_top = max(range(len(observed)), key=lambda k: observed[k])
    denominator = math.sqrt(sum(x*x for x in reference) * sum(x*x for x in observed))
    rms_error = math.sqrt(sum(x*x for x in errors) / len(errors))
    return dict(maxAbsoluteError=max(errors), rmsError=rms_error,
                cosineSimilarity=sum(a*b for a, b in zip(reference, observed)) / denominator if denominator else None,
                referenceTopToken=top[0], dopplerTopToken=observed_top,
                referenceMargin=reference[top[0]] - reference[top[1]],
                topReferenceTokens=[dict(token=k, reference=reference[k], doppler=observed[k], error=errors[k]) for k in top],
                passed=max(errors) <= tolerance and (rms_tolerance is None or rms_error <= rms_tolerance))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(reference_root, candidate_root, contract_path, out):
    ref_path, obs_path = reference_root / 'reference.json', candidate_root / 'observation.json'
    reference, observed = json.loads(ref_path.read_text()), json.loads(obs_path.read_text())
    contract = json.loads(contract_path.read_text())
    assert reference['manifestSha256'] == observed['manifestSha256']
    assert digest(contract_path) == reference['contractSha256']
    assert observed.get('executionCompleted') and not observed.get('error') and not observed.get('cleanupError')
    if 'contractSha256' in observed:
        assert observed['contractSha256'] == digest(contract_path)
        assert observed['inputCases'] == contract['cases']
    else:
        assert len(contract['cases']) == 1, 'Legacy receipt is single-case only'
    assert [c['id'] for c in reference['cases']] == [c['id'] for c in contract['cases']]
    tolerance = contract['acceptance']['implementationMaxAbsoluteLogitError']
    rows = []
    for case, inputs in zip(reference['cases'], contract['cases']):
        observations = [r for r in observed['rows'] if r.get('caseId', case['id']) == case['id']]
        assert case['positions'] == inputs['positions'] == [r['position'] for r in observations]
        reference_file = reference_root / case['logits']['path']
        assert digest(reference_file) == case['logits']['sha256']
        expected = read_npy(reference_file)
        assert len(expected) == len(observations)
        for values, row in zip(expected, observations):
            filename = candidate_root / row['file']
            assert digest(filename) == row['sha256']
            actual = struct.unpack('<' + str(row['count']) + 'f', filename.read_bytes())
            rows.append(dict(caseId=case['id'], position=row['position'], **compare(
                values, actual, tolerance, contract['acceptance'].get('implementationRmsLogitError'))))
    assert len(rows) == len(observed['rows'])
    states = []
    for item in reference.get('states', []):
        matches = [r for r in observed['states'] if all(r[k] == item[k] for k in ['caseId', 'layer', 'field', 'position'])]
        assert len(matches) == 1
        row = matches[0]
        a, b = reference_root / item['path'], candidate_root / row['file']
        assert digest(a) == item['sha256'] and digest(b) == row['sha256']
        shape, expected = read_tensor(a)
        expected_shape = ((1, row['numVHeads'], row['headKDim'], row['headVDim']) if item['field'] == 'recurrentState'
                          else (1, row['convDim'], row['convKernelSize']))
        assert shape == expected_shape == tuple(item['shape'])
        actual = struct.unpack('<' + str(row['count']) + 'f', b.read_bytes())
        comparison = compare(expected, actual, contract['acceptance']['implementationMaxAbsoluteStateError'],
                             contract['acceptance'].get('implementationRmsStateError'))
        states.append({**{k: item[k] for k in ['caseId', 'layer', 'field', 'position']},
                       **{k: comparison[k] for k in ['maxAbsoluteError', 'rmsError', 'cosineSimilarity', 'passed']}})
    assert len(states) == len(observed.get('states', [])) == len(contract.get('stateLayers', [])) * len(contract['cases']) * 2
    result = dict(schema='doppler.qwen-quantized-forced-logit-comparison/v1',
                  manifestSha256=reference['manifestSha256'], referenceSha256=digest(ref_path),
                  candidateSha256=digest(obs_path), contractSha256=digest(contract_path), toolSha256=digest(Path(__file__)),
                  maxAbsoluteLogitError=tolerance, acceptance=contract['acceptance'], rows=rows, states=states,
                  passed=all(r['passed'] for r in rows + states),
                  scope='Frozen shared input histories and declared state snapshots; not model quality or release qualification')
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
