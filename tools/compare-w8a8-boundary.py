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
    records = {}
    for row in result['captures']:
        previous = records.get(row['opId'])
        if previous is not None:
            assert compare_arrays(previous['capture']['data'], row['capture']['data'])['differentBits'] == 0, row['opId']
        records[row['opId']] = row
    if source.get('residuals'):
        report = {'schema': 'doppler.w8a8-residual-comparison/v1', 'modelIdentity': identity,
                  'gpuSha256': digest(args.gpu), 'sourceSha256': digest(args.source),
                  'context': result['context'], 'differentLogitBits': result['differentLogitBits'],
                  'residuals': {index: compare_arrays(values, records['layer.' + index + '.layer.out']['capture']['data'])
                                for index, values in source['residuals'].items()}}
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    layer = source['contract']['layer']
    prefix = f'layer.{layer}.'
    ports = {
        'q_proj': ('self_attn', 'attn.post_input_norm', 'attn.q_proj'),
        'k_proj': ('self_attn', 'attn.post_input_norm', 'attn.k_proj'),
        'v_proj': ('self_attn', 'attn.post_input_norm', 'attn.v_proj'),
        'o_proj': ('self_attn', 'attn.core_out', 'attn.out'),
        'gate_proj': ('mlp', 'ffn.in', 'ffn.gate'),
        'up_proj': ('mlp', 'ffn.in', 'ffn.up'),
        'down_proj': ('mlp', 'ffn.act', 'ffn.out'),
    }
    files = {row['file']: row['sha256'] for row in binding['files']}
    report = {'schema': 'doppler.w8a8-actual-operand-comparison/v1', 'modelIdentity': identity,
              'gpuSha256': digest(args.gpu), 'sourceSha256': digest(args.source),
              'bindingSha256': digest(args.binding), 'toolSha256': digest(__file__),
              'referenceSha256': digest(Path(__file__).parent / 'lib/w8a8_model_reference.py'),
              'scope': 'Actual GPU operands checked independently; source inputs remain a separate inherited-error control',
              'projections': {}}
    for name, independently_evolved in source['captured'].items():
        owner, input_port, output_port = ports[name]
        values = records[prefix + input_port]['capture']['data']
        inputs = np.asarray(values, np.float32).reshape(1, -1)
        input_codes, input_scales = quantize_rows(inputs)
        entry = manifest['tensors'][f'model.layers.{layer}.{owner}.{name}.weight']
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
        observed = records[prefix + output_port]['capture']['data']
        report['projections'][name] = {
            'actualOperandProjection': compare_arrays(expected, observed),
            'sourceInput': compare_arrays(independently_evolved['input'], inputs.flatten()),
            'activationCodesDifferent': int(np.count_nonzero(input_codes.flatten() != independently_evolved['codes'])),
            'activationScale': compare_arrays(independently_evolved['scales'], input_scales),
            'inheritedInputEffect': compare_arrays(independently_evolved['output'], expected),
        }
    if args.residual_gpu:
        prior = json.loads(Path(args.residual_gpu).read_text())
        assert prior['manifestDigest'] == identity and prior['result']['differentLogitBits'] == 0
        assert prior['result']['context'] == result['context']
        previous = {}
        for row in prior['result']['captures']:
            key = row['opId']
            if key in previous:
                assert compare_arrays(previous[key], row['capture']['data'])['differentBits'] == 0
            previous[key] = row['capture']['data']
        def values(port):
            return np.asarray(records[prefix + port]['capture']['data'], np.float64)
        def norm(operand, name):
            entry = manifest['tensors'][f'model.layers.{layer}.{name}.weight']
            assert entry['dtype'] == 'F16' and entry['shape'] == [640]
            filename = manifest['shards'][entry['shard']]['filename']
            assert digest(root / filename) == files[filename]
            raw = (root / filename).read_bytes()[entry['offset']:entry['offset'] + entry['size']]
            weights = np.frombuffer(raw, '<f2').astype(np.float64)
            eps = manifest['inference']['normalization']['rmsNormEps']
            return operand / np.sqrt(np.mean(operand ** 2) + eps) * (1 + weights)
        residual = np.asarray(previous[f'layer.{layer - 1}.layer.out'], np.float64)
        gate, up = values('ffn.gate'), values('ffn.up')
        activated = .5 * gate * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + .044715 * gate ** 3))) * up
        post_attn = values('attn.post_attn')
        report['nonlinearActualOperandsF64'] = {
            'reference': 'Independent binary64 equations rounded once for comparison; ordinary GPU F32 evaluation is not required bit-identical',
            'residualGpuSha256': digest(args.residual_gpu),
            'inputNorm': compare_arrays(norm(residual, 'input_layernorm'), values('attn.post_input_norm')),
            'attentionNormAndResidual': compare_arrays(norm(values('attn.out'), 'post_attention_layernorm') + residual, post_attn),
            'feedforwardNorm': compare_arrays(norm(post_attn, 'pre_feedforward_layernorm'), values('ffn.in')),
            'gatedGelu': compare_arrays(activated, values('ffn.act')),
            'feedforwardNormAndResidual': compare_arrays(norm(values('ffn.out'), 'post_feedforward_layernorm') + post_attn, values('layer.out')),
        }
        source_down = np.asarray(source['captured']['down_proj']['output'], np.float64)
        report['postFeedforwardAmplification'] = {
            'rawDownDifference': compare_arrays(source_down, values('ffn.out')),
            'postNormDifference': compare_arrays(norm(source_down, 'post_feedforward_layernorm'), norm(values('ffn.out'), 'post_feedforward_layernorm')),
        }
    report['actualOperandProjectionsExact'] = all(row['actualOperandProjection']['differentBits'] == 0 for row in report['projections'].values())
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['gpu', 'source', 'model', 'binding', 'out']:
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--residual-gpu')
    run(parser.parse_args())
