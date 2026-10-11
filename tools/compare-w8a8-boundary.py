#!/usr/bin/env python3
"""Compare captured operands using independent integer projection arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from lib.w8a8_model_reference import decode_rows, quantize_rows, IndependentW8A8Linear


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_arrays(expected, actual):
    a, b = np.asarray(expected, dtype=np.float32), np.asarray(actual, dtype=np.float32)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    different = a.view(np.uint32) != b.view(np.uint32)
    return {'values': int(a.size), 'differentBits': int(different.sum()),
            'maxAbs': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
            'firstDifference': int(np.flatnonzero(different)[0]) if different.any() else None}


def run(args):
    torch.set_num_threads(1)
    root = Path(args.model)
    gpu = json.loads(Path(args.gpu).read_text())
    source = json.loads(Path(args.source).read_text())
    binding = json.loads(Path(args.binding).read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = 'sha256:' + digest(root / 'manifest.json')
    assert identity == binding['preparedManifest'] == gpu['manifestDigest'] == source['modelIdentity']
    result = gpu['result']
    assert result['passed'] and result['differentLogitBits'] == 0
    assert result['context'] == source['contract']['inputIds'] == [2, 236776]
    records = {row['opId']: row for row in result['captures']}
    values = records['layer.0.attn.post_input_norm']['capture']['data']
    inputs = np.asarray(values, np.float32).reshape(1, 640)
    input_codes, input_scales = quantize_rows(inputs)
    files = {row['file']: row['sha256'] for row in binding['files']}
    report = {'schema': 'doppler.w8a8-actual-operand-comparison/v1', 'modelIdentity': identity,
              'gpuSha256': digest(args.gpu), 'sourceSha256': digest(args.source),
              'bindingSha256': digest(args.binding), 'toolSha256': digest(__file__),
              'referenceSha256': digest(Path(__file__).parent / 'lib/w8a8_model_reference.py'),
              'scope': 'Actual GPU operands checked independently; source inputs remain a separate inherited-error control',
              'projections': {}}
    for name in ['q_proj', 'k_proj', 'v_proj']:
        entry = manifest['tensors']['model.layers.0.self_attn.' + name + '.weight']
        spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
        buffers = []
        for span in spans:
            filename = manifest['shards'][span['shardIndex']]['filename']
            assert digest(root / filename) == files[filename]
            buffers.append((root / filename).read_bytes()[span['offset']:span['offset'] + span['size']])
        codes, scales = decode_rows(b''.join(buffers), entry)
        module = IndependentW8A8Linear(codes, scales)
        with torch.inference_mode():
            expected = module(torch.from_numpy(inputs)).flatten().numpy()
        observed = records['layer.0.attn.' + name]['capture']['data']
        independently_evolved = source['captured'][name]
        report['projections'][name] = {
            'actualOperandProjection': compare_arrays(expected, observed),
            'sourceInput': compare_arrays(independently_evolved['input'], inputs.flatten()),
            'activationCodesDifferent': int(np.count_nonzero(input_codes.flatten() != independently_evolved['codes'])),
            'activationScale': compare_arrays(independently_evolved['scales'], input_scales),
            'inheritedInputEffect': compare_arrays(independently_evolved['output'], expected),
        }
    report['actualOperandProjectionsExact'] = all(row['actualOperandProjection']['differentBits'] == 0 for row in report['projections'].values())
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['gpu', 'source', 'model', 'binding', 'out']:
        parser.add_argument('--' + field, required=True)
    run(parser.parse_args())
