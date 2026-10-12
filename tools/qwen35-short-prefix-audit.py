#!/usr/bin/env python3
"""Observe the approved evaluator without editing it; audit request 0, step 1.

CPU diagnostic only. Captured GPU operands enter local operation comparisons
after independent evaluation, never the authoritative reference state.
"""
import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


EVALUATOR_SHA256 = '90376ddec6e6bf997425ee0e1ecf1d98533434ebbe3f934ff00a878dd49e8bc7'


def error(actual, expected):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = np.abs(actual - expected)
    return {'values': int(difference.size), 'max': float(difference.max()),
            'rms': float(np.sqrt(np.mean(difference ** 2))),
            'percentiles': dict(zip(['50', '90', '99', '99.9'],
                                    map(float, np.percentile(difference, [50, 90, 99, 99.9])))),
            'overTolerance': int(np.count_nonzero(difference > 0.001)),
            'lastTokenMax': float(difference[-1].max())}


def read_capture(capture):
    """Pair by actual generation step and operation; reject conflicting duplicates."""
    groups, step = {}, -1
    for row in capture['observation']['timeline']:
        if row['opId'] == 'embed.out':
            step += 1
        item = row.get('capture')
        if not item or item.get('level') != 'full':
            continue
        assert item['encoding'] == 'base64-f32le' and step in (0, 1)
        value = np.frombuffer(base64.b64decode(item['dataBase64']), dtype='<f4')
        value = value.reshape(item['shape']).astype(np.float64)
        key = (row['opId'], step)
        if key in groups:
            assert np.array_equal(groups[key], value), key
        groups[key] = value
    result = {}
    for name in sorted({name for name, _ in groups}):
        # Fused decode does not expose the prefill-only FFN gate/up tensors.
        # Required paired operations are accessed explicitly by the audit below.
        if (name, 0) not in groups or (name, 1) not in groups:
            assert name in ('layer.0.ffn.gate', 'layer.0.ffn.up'), name
            continue
        first, second = groups[name, 0], groups[name, 1]
        assert first.shape[0] == 19 and second.shape[0] == 1, name
        result[name] = np.concatenate([first, second])
    return result


def run(args):
    source = Path(__file__).with_name('qwen35-f64-reference.py')
    spec = importlib.util.spec_from_file_location('approved_f64', source)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    assert ref.digest(source.read_bytes()) == EVALUATOR_SHA256
    contract = ref.read_json('artifacts/whole-model-f64-complete-20261011/request-0-contract.json')
    for name in ('manifest', 'sourceConfig', 'pieceIndex'):
        item = contract['inputs'][name]
        assert ref.digest(Path(item['path']).read_bytes()) == item['sha256'], name
    config = ref.read_json(contract['inputs']['sourceConfig']['path'])['text_config']
    manifest_path = Path(contract['inputs']['manifest']['path'])
    weights, identities = ref.load_weights(manifest_path.parent, ref.read_json(manifest_path),
                                          ref.read_json(contract['inputs']['pieceIndex']['path']))
    independent = {}
    def retain(name, value):
        independent[name] = value.reshape(len(contract['inputIds']), -1).copy()
    # Line events observe locals before the statement executes. The entire
    # evaluator source is pinned above; no function or local variable is replaced.
    def trace(frame, event, _arg):
        if frame.f_code is not ref.evaluate.__code__:
            return None
        if event != 'line':
            return trace
        local, line = frame.f_locals, frame.f_lineno
        if line == 101:
            retain('embed.out', local['x'])
        elif line == 107:
            retain(f'layer.{local["index"]}.attn.post_input_norm', local['normalized'])
        elif line == 139:
            prefix = f'layer.{local["index"]}.attn.'
            retain(prefix + 'out', local['update'])
            if local['kind'] == 'linear_attention':
                retain(prefix + 'linear_core_out', local['core'])
        elif line == 144:
            retain(f'layer.{local["index"]}.layer.out', local['x'])
        elif line == 149:
            independent['final_norm.out'] = local['normalized'].copy()
        return trace
    def observe_cache(layer, values):
        for name, value in values.items():
            retain(f'layer.{layer}.attn.{name}', value)
    assert sys.gettrace() is None
    sys.settrace(trace)
    try:
        logits, caches = ref.evaluate(contract['inputIds'], weights, config,
            np.array(contract['prefixLengths']) - 1,
            lambda row: print(json.dumps(row), flush=True), observe_cache)
    finally:
        sys.settrace(None)
    approved = ref.read_json(args.approved)
    assert approved['toolSha256'] == EVALUATOR_SHA256
    expected = np.stack([np.frombuffer(base64.b64decode(row['logitsF64']), dtype='<f8')
                         for row in approved['results']])
    assert np.array_equal(logits, expected), 'Observation must exactly reproduce approved vectors'
    capture = ref.read_json(args.capture)
    assert capture['observationTarget'] == {'index': 0, 'step': 1, 'physicalCache': True}
    assert capture['observation']['maximumUnmaskedDifference'] == 0
    assert capture['manifestIdentity'] == 'sha256:' + contract['inputs']['manifest']['sha256']
    gpu = read_capture(capture)
    if args.fixture_directory:
        directory = Path(args.fixture_directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = ref.read_json(manifest_path)
        for layer, operation, tensor in [(0, 'qkv_proj', 'linear_attn.in_proj_qkv.weight'),
                                         (3, 'k_proj', 'self_attn.k_proj.weight'),
                                         (3, 'v_proj', 'self_attn.v_proj.weight')]:
            name = f'layers.{layer}.{tensor}'
            entry = manifest['tensors']['model.language_model.' + name]
            assert entry['dtype'] == 'Q4_K_M' and entry['layout'] == 'row'
            spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
            raw = b''.join((manifest_path.parent / manifest['shards'][s['shardIndex']]['filename']).read_bytes()
                          [s['offset']:s['offset'] + s['size']] for s in spans)
            actual_input = gpu[f'layer.{layer}.attn.post_input_norm'][:19]
            actual_output = gpu[f'layer.{layer}.attn.{operation}'][:19]
            input_bytes = actual_input.astype('<f4').tobytes()
            encode = lambda value: base64.b64encode(value).decode()
            fixture = {'M': 19, 'N': entry['shape'][0], 'K': entry['shape'][1],
                'operation': f'layer.{layer}.attn.{operation}', 'input': encode(input_bytes),
                'inputSha256': ref.digest(input_bytes), 'weights': encode(raw), 'weightsSha256': ref.digest(raw),
                'oracleFloat64': encode((actual_input @ weights[name].T).astype('<f8').tobytes()),
                'baseline026': encode(actual_output.astype('<f4').tobytes()),
                'captureSha256': ref.digest(Path(args.capture).read_bytes())}
            (directory / f'layer-{layer}-{operation}.json').write_text(json.dumps(fixture) + '\n')
    rows, storage = [], []
    def compare(name, local=None):
        actual = gpu[name]
        row = {'operation': name}
        if name in independent:
            row['total'] = error(actual, independent[name])
        if local is not None:
            row['local'] = error(actual, local.reshape(actual.shape))
            if name in independent:
                row['inherited'] = error(local.reshape(actual.shape), independent[name])
        rows.append(row)
    compare('embed.out')
    for layer, kind in enumerate(config['layer_types']):
        prefix = f'layer.{layer}.attn.'
        w = lambda name: weights[f'layers.{layer}.' + name]
        before = gpu['embed.out' if layer == 0 else f'layer.{layer - 1}.layer.out']
        compare(prefix + 'post_input_norm', ref.rms_norm(before, w('input_layernorm.weight'), config['rms_norm_eps']))
        normalized = gpu[prefix + 'post_input_norm']
        if kind == 'linear_attention':
            for output, tensor in [('qkv_proj', 'qkv'), ('linear_z_proj', 'z'),
                                   ('linear_a_proj', 'a'), ('linear_b_proj', 'b')]:
                matrix = w(f'linear_attn.in_proj_{tensor}.weight').T
                independent[prefix + output] = independent[prefix + 'post_input_norm'] @ matrix
                compare(prefix + output, normalized @ matrix)
            core = ref.delta_history(*[gpu[prefix + name] for name in
                ('qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj')],
                w('linear_attn.conv1d.weight').squeeze(1), w('linear_attn.A_log'),
                w('linear_attn.dt_bias'), w('linear_attn.norm.weight'), config)
            compare(prefix + 'linear_core_out', core)
            compare(prefix + 'out', gpu[prefix + 'linear_core_out'] @ w('linear_attn.out_proj.weight').T)
        else:
            nh, nk, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
            for output, tensor in [('k_proj', 'k'), ('v_proj', 'v')]:
                compare(prefix + output, normalized @ w(f'self_attn.{tensor}_proj.weight').T)
            key = ref.rms_norm(gpu[prefix + 'k_proj'].reshape(-1, nk, width),
                               w('self_attn.k_norm.weight'), config['rms_norm_eps'])
            compare(prefix + 'k_norm', key)
            rope = config['rope_parameters']
            compare(prefix + 'k_rope', ref.text_rope(gpu[prefix + 'k_norm'].reshape(-1, nk, width),
                rope['rope_theta'], int(width * rope['partial_rotary_factor'])))
            # Recompute the complete producer on the captured normalized input:
            # projection, K norm/RoPE, then the unchanged F16 store. Differences
            # still present here were inherited before this producer boundary.
            local_key = ref.rms_norm((normalized @ w('self_attn.k_proj.weight').T).reshape(-1, nk, width),
                w('self_attn.k_norm.weight'), config['rms_norm_eps'])
            producer = {'k_rope': ref.text_rope(local_key, rope['rope_theta'],
                int(width * rope['partial_rotary_factor'])).reshape(len(normalized), -1),
                'v_proj': normalized @ w('self_attn.v_proj.weight').T}
            for name in ('k_rope', 'v_proj'):
                a, b = gpu[prefix + name], independent[prefix + name]
                ah, bh = ref.half_cache(a), ref.half_cache(b)
                local = producer[name]
                lh = ref.half_cache(local)
                coordinates = np.argwhere(ah != bh)
                storage.append({'operation': prefix + name, 'before': error(a, b), 'after': error(ah, bh),
                    'producerLocalBefore': error(a, local), 'producerInheritedBefore': error(local, b),
                    'producerIntroducedStoredDifferences': int(np.count_nonzero((ah != bh) & (lh == bh))),
                    'inheritedStoredDifferencesRemaining': int(np.count_nonzero((ah != bh) & (lh != bh))),
                    'producerCancelledInheritedStoredDifferences': int(np.count_nonzero((ah == bh) & (lh != bh))),
                    'differentValues': len(coordinates), 'coordinates': [
                        {'tokenDimension': list(map(int, c)), 'gpu': float(a[tuple(c)]),
                         'f64': float(b[tuple(c)]), 'gpuStored': float(ah[tuple(c)]),
                         'sameInputProducerF64': float(local[tuple(c)]), 'sameInputProducerStored': float(lh[tuple(c)]),
                         'f64Stored': float(bh[tuple(c)])} for c in coordinates]})
            # core_out precedes the attention output gate in the runtime.
            query = gpu[prefix + 'q_rope'].reshape(-1, nh, width)
            key = ref.half_cache(gpu[prefix + 'k_rope'].reshape(-1, nk, width))
            value = ref.half_cache(gpu[prefix + 'v_proj'].reshape(-1, nk, width))
            compare(prefix + 'core_out', ref.causal_attention(query, key, value))
            compare(prefix + 'out')
        compare(f'layer.{layer}.layer.out')
    gpu_norm = gpu['final_norm.out'][np.array(contract['prefixLengths']) - 1]
    rows.append({'operation': 'final_norm.out', 'total': error(gpu_norm, independent['final_norm.out']),
                 'local': error(gpu_norm, ref.rms_norm(gpu['layer.23.layer.out'][[18, 19]],
                                                     weights['norm.weight'], config['rms_norm_eps']))})
    actual_logits = np.stack([np.frombuffer(base64.b64decode(row['logits']), dtype='<f4').astype(np.float64)
                              for row in capture['results']])
    local_logits = gpu_norm @ weights['lm_head.weight'].T
    report = {'schema': 'doppler.short-prefix-f64-audit/v1', 'qualified': False,
        'scope': 'Independent-state total error versus same-GPU-operands local error; diagnostic only',
        'evaluatorSha256': EVALUATOR_SHA256, 'auditSha256': ref.digest(Path(__file__).read_bytes()),
        'captureSha256': ref.digest(Path(args.capture).read_bytes()),
        'approvedVectorsSha256': ref.digest(Path(args.approved).read_bytes()),
        'approvedVectorsExact': True, 'shards': identities, 'cacheIdentities': caches,
        'operations': rows, 'storage': storage, 'logits': {
            'total': error(actual_logits, logits), 'local': error(actual_logits, local_logits),
            'inherited': error(local_logits, logits), 'firstFailure': {
                'index': 0, 'step': 1, 'vocabulary': 9072, 'gpu': float(actual_logits[1, 9072]),
                'f64': float(logits[1, 9072]), 'localF64': float(local_logits[1, 9072])}}}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--approved', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fixture-directory', help='Export captured prefill projections for the existing physical replay')
    run(parser.parse_args())
