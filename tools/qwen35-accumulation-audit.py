#!/usr/bin/env python3
"""Retained actual-operand audit of F32 Q4 projection accumulation accuracy."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('f64_reference', Path(__file__).with_name('qwen35-f64-reference.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def accumulate(x, weight, mode, lanes=4):
    """Emulate strided F32 lane order, including explicit F32 rounding.

    FMA emulation uses a binary64 product/sum followed by F32 rounding. The
    compensated candidate uses only F32 products/additions/subtractions.
    This does not assert how a particular driver optimizes WGSL expressions.
    """
    assert x.dtype == weight.dtype == np.float32 and x.shape[1] == weight.shape[1]
    if mode in ('q4-widetile', 'q4-widetile-blocked'):
        assert x.shape[1] % 256 == 0
        # Q4 pairs interleave low/high nibbles, four elements from each.
        order = [base + offset + lane for base in range(0, x.shape[1], 64)
                 for group in range(8) for offset in (group * 4, 32 + group * 4)
                 for lane in range(4)]
        x, weight = x[:, order], weight[:, order]
        if mode == 'q4-widetile':
            return accumulate(x, weight, 'fma', 4)
        total = np.zeros((len(x), len(weight)), dtype=np.float32)
        for start in range(0, x.shape[1], 256):
            total += accumulate(x[:, start:start + 256], weight[:, start:start + 256], 'fma', 4)
        return total
    if mode == 'blocked-tile':
        assert lanes > 0
        total = np.zeros((len(x), len(weight)), dtype=np.float32)
        for start in range(0, x.shape[1], lanes):
            total += accumulate(x[:, start:start + lanes], weight[:, start:start + lanes], 'fma', 1)
        return total
    sums = np.zeros((len(x), len(weight), lanes), dtype=np.float32)
    corrections = np.zeros_like(sums)
    for k in range(x.shape[1]):
        lane = k % lanes
        current = sums[:, :, lane]
        if mode in ('fma', 'tiled-pairwise'):
            sums[:, :, lane] = (x[:, k, None].astype(np.float64) * weight[None, :, k].astype(np.float64)
                                + current.astype(np.float64)).astype(np.float32)
        else:
            product = x[:, k, None] * weight[None, :, k]
            if mode == 'compensated':
                adjusted = product - corrections[:, :, lane]
                total = current + adjusted
                corrections[:, :, lane] = (total - current) - adjusted
                sums[:, :, lane] = total
            else:
                assert mode == 'separate'
                sums[:, :, lane] += product
    if mode == 'tiled-pairwise':
        assert lanes > 0 and lanes & (lanes - 1) == 0
        stride = lanes // 2
        while stride:
            sums[:, :, :stride] += sums[:, :, stride:2 * stride]
            stride //= 2
        return sums[:, :, 0]
    total = np.zeros(sums.shape[:2], dtype=np.float32)
    correction = np.zeros_like(total)
    for lane in range(lanes):
        term = sums[:, :, lane]
        if mode == 'compensated':
            term = term - correction
            updated = total + term
            correction = (updated - total) - term
            total = updated
        else:
            total += term
    return total


def error(actual, precise):
    delta = actual.astype(np.float64) - precise
    return {'maxAbsError': float(np.max(np.abs(delta))), 'rmsError': float(np.sqrt(np.mean(delta * delta)))}


def run(args):
    contract = reference.read_json(args.contract)
    inputs = contract['inputs']
    for name in ('manifest', 'pieceIndex', 'sourceConfig'):
        assert reference.digest(Path(inputs[name]['path']).read_bytes()) == inputs[name]['sha256']
    manifest = reference.read_json(inputs['manifest']['path'])
    config = reference.read_json(inputs['sourceConfig']['path'])['text_config']
    weights, _ = reference.load_weights(Path(inputs['manifest']['path']).parent, manifest,
                                         reference.read_json(inputs['pieceIndex']['path']))
    capture = reference.read_json(args.capture)
    assert capture['manifestIdentity'] == 'sha256:' + inputs['manifest']['sha256']
    selected = [0, 1, 2, 15, 31, 63, 127, 183]
    def observed(name, positions=selected):
        row = next(row for row in capture['observation']['timeline'] if row['opId'] == name and row.get('capture'))
        values = row['capture']
        assert values['encoding'] == 'base64-f32le'
        result = np.frombuffer(base64.b64decode(values['dataBase64']), dtype='<f4').reshape(184, -1)
        return result[positions]
    embedding = observed('embed.out')
    assert np.array_equal(embedding, weights['embed_tokens.weight'][np.array(contract['inputIds'])[selected]])
    normalized = observed('layer.0.attn.post_input_norm')
    precise_norm = reference.rms_norm(embedding.astype(np.float64), weights['layers.0.input_layernorm.weight'], config['rms_norm_eps'])
    results = []
    for boundary, weight_name in [('qkv_proj', 'in_proj_qkv'), ('linear_z_proj', 'in_proj_z'),
                                  ('linear_a_proj', 'in_proj_a'), ('linear_b_proj', 'in_proj_b')]:
        tensor_name = f'layers.0.linear_attn.{weight_name}.weight'
        weight = weights[tensor_name].astype(np.float32)
        storage = manifest['tensors']['model.language_model.' + tensor_name]['dtype']
        assert storage in ('Q4_K_M', 'F16')
        lanes = 4 if storage == 'Q4_K_M' else 1
        gpu = observed('layer.0.attn.' + boundary)
        precise = normalized.astype(np.float64) @ weight.astype(np.float64).T
        global_precise = precise_norm @ weight.astype(np.float64).T
        modes = {}
        for mode in ('separate', 'fma', 'compensated'):
            actual = accumulate(normalized, weight, mode, lanes)
            modes[mode] = {**error(actual, precise), 'versusGpu': error(actual, gpu),
                           'equalGpuValues': int(np.sum(actual == gpu)), 'values': int(actual.size)}
        if storage == 'Q4_K_M':
            for mode in ('q4-widetile', 'q4-widetile-blocked'):
                actual = accumulate(normalized, weight, mode)
                modes[mode] = {**error(actual, precise), 'versusGpu': error(actual, gpu),
                               'equalGpuValues': int(np.sum(actual == gpu)), 'values': int(actual.size)}
        if storage == 'F16':
            actual = accumulate(normalized, weight, 'blocked-tile', 16)
            modes['blocked-tile'] = {**error(actual, precise), 'versusGpu': error(actual, gpu),
                                    'values': int(actual.size), 'tileTerms': 16}
        row = {'boundary': boundary, 'weightStorage': storage, 'accumulationLanes': lanes,
               'gpuLocalError': error(gpu, precise),
               'inheritedInputEffect': error(precise, global_precise), 'gpuTotalError': error(gpu, global_precise),
               'minimumF32RoundingError': error(precise.astype(np.float32), precise), 'modes': modes}
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {'schema': 'doppler.qwen35-f32-accumulation-audit/v1', 'qualified': False,
              'scope': 'Selected actual prefill operands, layer0; no runtime changes or application-speed claim',
              'contractSha256': reference.digest(Path(args.contract).read_bytes()),
              'captureSha256': reference.digest(Path(args.capture).read_bytes()),
              'toolSha256': reference.digest(Path(__file__).read_bytes()), 'selectedPositions': selected,
              'normalizationError': error(normalized, precise_norm), 'embeddingExact': True, 'results': results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    if args.q4_operands_out:
        encode = lambda values, dtype: base64.b64encode(values.astype(dtype).tobytes()).decode()
        root = Path(inputs['manifest']['path']).parent
        def raw_weight(name):
            entry = manifest['tensors']['model.language_model.' + name]
            spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
            raw = b''.join((root / manifest['shards'][s['shardIndex']]['filename']).read_bytes()[s['offset']:s['offset'] + s['size']] for s in spans)
            return np.frombuffer(raw, dtype=np.uint8).reshape(entry['shape'][0], -1, 144)
        cases = []
        for name, boundary in [('qkv', 'qkv_proj'), ('z', 'linear_z_proj')]:
            tensor = f'layers.0.linear_attn.in_proj_{name}.weight'
            weight, raw = weights[tensor], raw_weight(tensor)
            cases.append({'name': boundary, 'kind': 'actual', 'M': len(selected), 'N': len(weight), 'K': 1024,
                          'alpha': 1, 'inputF32': encode(normalized, '<f4'), 'weightQ4': encode(raw, 'u1'),
                          'expectedF64': encode(normalized.astype(np.float64) @ weight.T, '<f8'),
                          'observedF32': encode(observed('layer.0.attn.' + boundary), '<f4')})
        raw = raw_weight('layers.0.linear_attn.in_proj_qkv.weight')
        for index, (m, n, k, alpha) in enumerate([(1, 1, 256, 1), (5, 257, 1024, -0.5),
                                                (9, 17, 768, 1), (3, 3, 257, 1), (8, 65, 4096, 2)]):
            blocks = (k + 255) // 256
            packed = np.tile(raw[:n], (1, (blocks + 3) // 4, 1))[:, :blocks].copy()
            weight = reference.dequantize(packed.reshape(-1), reference.GGMLQuantizationType.Q4_K).reshape(n, -1)[:, :k].astype(np.float64)
            x = (((np.arange(m * k).reshape(m, k) * 17 + 13) % 113) - 56).astype(np.float32) / 16
            expected = (x.astype(np.float64) @ weight.T) * alpha
            bounds = np.abs(x.astype(np.float64)) @ np.abs(weight.T) * abs(alpha) * (k + 8) * 2 ** -23 + 1e-10
            cases.append({'name': f'shape_{index}', 'kind': 'synthetic', 'M': m, 'N': n, 'K': k, 'alpha': alpha,
                          'tileM': 8 if index == 2 else 4, 'tileN': 128 if index == 2 else 256,
                          'inputF32': encode(x, '<f4'), 'weightQ4': encode(packed, 'u1'),
                          'expectedF64': encode(expected, '<f8'), 'errorBoundF64': encode(bounds, '<f8')})
        Path(args.q4_operands_out).write_text(json.dumps({'schema': 'doppler.q4-projection-actual-operands/v1',
                                                        'captureSha256': report['captureSha256'], 'cases': cases}) + '\n')
    if args.operands_out:
        full_input = observed('layer.0.attn.post_input_norm', list(range(184)))
        encode = lambda values, dtype: base64.b64encode(values.astype(dtype).tobytes()).decode()
        cases = []
        for name in ('a', 'b'):
            weight = weights[f'layers.0.linear_attn.in_proj_{name}.weight']
            cases.append({'name': f'linear_{name}_proj', 'M': 184, 'N': 16, 'K': 1024,
                          'inputF32': encode(full_input, '<f4'), 'weightF16': encode(weight, '<f2'),
                          'expectedF64': encode(full_input.astype(np.float64) @ weight.T, '<f8'),
                          'observedF32': encode(observed(f'layer.0.attn.linear_{name}_proj', list(range(184))), '<f4')})
        Path(args.operands_out).write_text(json.dumps({'schema': 'doppler.f16-projection-actual-operands/v1',
                                                      'captureSha256': report['captureSha256'], 'cases': cases}) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--operands-out')
    parser.add_argument('--q4-operands-out')
    run(parser.parse_args())
