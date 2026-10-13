#!/usr/bin/env python3
"""Offline Qwen3.5 deployed-weight F64 diagnostic, never runtime computation.

The frozen contract owns inputs and evaluation policy. Only token IDs and decoded
weights enter the model; retained implementation outputs enter comparison after
independent evaluation. This tool does not activate an acceptance reference.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, dequantize


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def rms_norm(x, weight, epsilon, offset=True):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * (weight + int(offset))


def half_cache(x):
    result = x.astype('<f2').astype(np.float64)
    assert np.isfinite(result).all(), 'nonfinite F16 cache'
    return result


def text_rope(x, theta, rotary_dim):
    assert rotary_dim % 2 == 0 and rotary_dim <= x.shape[-1]
    frequencies = theta ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    angle = np.arange(len(x), dtype=np.float64)[:, None, None] * frequencies[None, None, :]
    first, second = np.split(x[..., :rotary_dim], 2, axis=-1)
    return np.concatenate((first * np.cos(angle) - second * np.sin(angle),
                           second * np.cos(angle) + first * np.sin(angle), x[..., rotary_dim:]), axis=-1)


def causal_attention(query, key, value):
    """[token, head, dimension] inputs; retained KV already rounded by caller."""
    assert query.shape[1] % key.shape[1] == 0 and key.shape == value.shape
    repeats = query.shape[1] // key.shape[1]
    key, value = np.repeat(key, repeats, axis=1), np.repeat(value, repeats, axis=1)
    scores = query.transpose(1, 0, 2) @ key.transpose(1, 2, 0) / np.sqrt(query.shape[-1])
    scores[:, np.triu_indices(len(query), 1)[0], np.triu_indices(len(query), 1)[1]] = -np.inf
    probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    return (probability @ value.transpose(1, 0, 2)).transpose(1, 0, 2)


def delta_history(qkv, z, a, b, conv_weight, a_log, dt_bias, norm_weight, config):
    nk, nv = config['linear_num_key_heads'], config['linear_num_value_heads']
    dk, dv = config['linear_key_head_dim'], config['linear_value_head_dim']
    assert nv % nk == 0
    conv_state = np.zeros_like(conv_weight)
    state = np.zeros((nv, dk, dv), dtype=np.float64)
    outputs = []
    for index, projected in enumerate(qkv):
        conv_state[:, :-1] = conv_state[:, 1:]
        conv_state[:, -1] = projected
        mixed = np.sum(conv_state * conv_weight, axis=-1)
        mixed *= sigmoid(mixed)
        query, key, value = np.split(mixed, [nk * dk, 2 * nk * dk])
        query, key = query.reshape(nk, dk), key.reshape(nk, dk)
        value = value.reshape(nv, dv)
        query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
        query = np.repeat(query, nv // nk, axis=0) / np.sqrt(dk)
        key = np.repeat(key, nv // nk, axis=0)
        decay = np.exp(-np.exp(a_log) * np.logaddexp(0, a[index] + dt_bias))
        state *= decay[:, None, None]
        memory = np.sum(state * key[:, :, None], axis=1)
        delta = (value - memory) * sigmoid(b[index])[:, None]
        state += key[:, :, None] * delta[:, None, :]
        output = np.sum(state * query[:, :, None], axis=1)
        gate = z[index].reshape(nv, dv)
        output = rms_norm(output, norm_weight, config['rms_norm_eps'], offset=False)
        outputs.append((output * gate * sigmoid(gate)).reshape(-1))
    return np.stack(outputs)


def evaluate(token_ids, weights, config, positions, progress=None, observe_cache=None):
    """Independent causal model; no observed implementation state is accepted."""
    assert config['hidden_act'] == 'silu' and not config['attention_bias']
    assert config['rope_parameters']['rope_type'] == 'default'
    assert config['attn_output_gate'] and config['attention_dropout'] == 0
    x = weights['embed_tokens.weight'][token_ids].copy()
    epsilon = config['rms_norm_eps']
    caches = []
    for index, kind in enumerate(config['layer_types']):
        prefix = f'layers.{index}.'
        w = lambda name: weights[prefix + name]
        normalized = rms_norm(x, w('input_layernorm.weight'), epsilon)
        if kind == 'linear_attention':
            local = lambda name: w('linear_attn.' + name)
            core = delta_history(normalized @ local('in_proj_qkv.weight').T,
                                 normalized @ local('in_proj_z.weight').T,
                                 normalized @ local('in_proj_a.weight').T,
                                 normalized @ local('in_proj_b.weight').T,
                                 local('conv1d.weight').squeeze(1), local('A_log'),
                                 local('dt_bias'), local('norm.weight'), config)
            update = core @ local('out_proj.weight').T
        else:
            assert kind == 'full_attention'
            local = lambda name: w('self_attn.' + name)
            heads, kv_heads, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
            query, gate = np.split((normalized @ local('q_proj.weight').T).reshape(len(x), heads, 2 * width), 2, axis=-1)
            query = rms_norm(query, local('q_norm.weight'), epsilon)
            key_projected = (normalized @ local('k_proj.weight').T).reshape(len(x), kv_heads, width)
            key_normalized = rms_norm(key_projected, local('k_norm.weight'), epsilon)
            value = (normalized @ local('v_proj.weight').T).reshape(len(x), kv_heads, width)
            rope = config['rope_parameters']
            dimension = int(width * rope['partial_rotary_factor'])
            query = text_rope(query, rope['rope_theta'], dimension)
            key_rotated = text_rope(key_normalized, rope['rope_theta'], dimension)
            if observe_cache:
                observe_cache(index, {'post_input_norm': normalized.copy(), 'k_proj': key_projected.copy(),
                                      'k_norm': key_normalized.copy(), 'k_rope': key_rotated.copy(),
                                      'v_proj': value.copy()})
            key = half_cache(key_rotated)
            value = half_cache(value)
            caches.append({'layer': index, 'keySha256': digest(key.astype('<f2').tobytes()),
                           'valueSha256': digest(value.astype('<f2').tobytes())})
            core = causal_attention(query, key, value) * sigmoid(gate)
            update = core.reshape(len(x), -1) @ local('o_proj.weight').T
        x += update
        normalized = rms_norm(x, w('post_attention_layernorm.weight'), epsilon)
        gate = normalized @ w('mlp.gate_proj.weight').T
        up = normalized @ w('mlp.up_proj.weight').T
        x += (gate * sigmoid(gate) * up) @ w('mlp.down_proj.weight').T
        assert np.isfinite(x).all(), f'nonfinite layer {index}'
        if progress:
            progress({'layer': index, 'kind': kind, 'hiddenSha256': digest(x.astype('<f8').tobytes()),
                      'maxAbsHidden': float(np.max(np.abs(x)))})
    normalized = rms_norm(x[positions], weights['norm.weight'], epsilon)
    return normalized @ weights['lm_head.weight'].T, caches


def load_weights(root, manifest, pieces):
    files = {row['path']: row for row in pieces['files']}
    shards, identities, weights = {}, [], {}
    for shard_index, shard in enumerate(manifest['shards']):
        data = (root / shard['filename']).read_bytes()
        assert len(data) == shard['size'] == files[shard['filename']]['size']
        cursor = 0
        for piece in files[shard['filename']]['pieces']:
            assert piece['offset'] == cursor
            assert 'sha256:' + digest(data[cursor:cursor + piece['size']]) == piece['identity']
            cursor += piece['size']
        assert cursor == len(data)
        identities.append({'path': shard['filename'], 'sha256': digest(data)})
        shards[shard_index] = data
    for name, entry in manifest['tensors'].items():
        if not name.startswith('model.language_model.'):
            continue
        name = name.removeprefix('model.language_model.')
        spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
        data = b''.join(shards[s['shardIndex']][s['offset']:s['offset'] + s['size']] for s in spans)
        assert len(data) == entry['size']
        if entry['dtype'] == 'Q4_K_M':
            assert entry['layout'] == 'row'
            array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
        else:
            assert entry['dtype'] in ('F16', 'F32')
            array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4')
        weights[name] = array.reshape(entry['shape']).astype(np.float64)
    return weights, identities


def comparison(actual, encoded, tolerance):
    other = np.frombuffer(base64.b64decode(encoded), dtype='<f4').astype(np.float64)
    assert other.shape == actual.shape and np.isfinite(other).all()
    difference = np.abs(actual - other)
    return {'maxAbsDifference': float(difference.max()), 'rmsDifference': float(np.sqrt(np.mean(difference ** 2))),
            'valuesOverTolerance': int(np.sum(difference > tolerance)), 'pass': bool(np.all(difference <= tolerance)),
            'argmaxDifference': int(np.argmax(difference)), 'greedyToken': int(np.argmax(other))}


def compare_first_cache(observations, capture, config):
    layer = config['layer_types'].index('full_attention')
    checks = []
    for name, actual in observations.items():
        parts = [row['capture'] for row in capture['observation']['timeline']
                 if row['opId'] == f'layer.{layer}.attn.{name}']
        assert parts and all(row['encoding'] == 'base64-f32le' for row in parts)
        gpu = np.concatenate([np.frombuffer(base64.b64decode(row['dataBase64']), dtype='<f4') for row in parts]).reshape(actual.shape).astype(np.float64)
        difference = np.abs(gpu - actual)
        result = {'boundary': name, 'values': int(actual.size), 'maxAbsDifference': float(difference.max())}
        if name in ('k_rope', 'v_proj'):
            gpu_half, actual_half = half_cache(gpu), half_cache(actual)
            coordinates = np.argwhere(gpu_half != actual_half)
            result['differentStoredValues'] = len(coordinates)
            if len(coordinates):
                coordinate = tuple(coordinates[0])
                result['firstStorageDifference'] = {'positionHeadDimension': [int(x) for x in coordinate],
                                                    'gpuBeforeStore': float(gpu[coordinate]),
                                                    'independentF64BeforeStore': float(actual[coordinate]),
                                                    'gpuStored': float(gpu_half[coordinate]),
                                                    'independentF64Stored': float(actual_half[coordinate])}
        checks.append(result)
    return {'layer': layer, 'checks': checks,
            'scope': 'Post-computation comparison only; GPU operands never enter the independently evolved model'}


def run(contract_path, output, boundary_capture=None):
    contract = read_json(contract_path)
    assert contract['schema'] == 'doppler.qwen35-independent-f64-contract/v1'
    assert contract['tolerance'] == 0.001
    inputs = contract['inputs']
    for name, entry in inputs.items():
        assert digest(Path(entry['path']).read_bytes()) == entry['sha256'], f'{name} identity changed'
    manifest = read_json(inputs['manifest']['path'])
    config = read_json(inputs['sourceConfig']['path'])['text_config']
    assert config['hidden_size'] == manifest['architecture']['hiddenSize']
    assert config['layer_types'] == manifest['inference']['layerPattern']['layerTypes']
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    root = Path(inputs['manifest']['path']).parent
    weights, identities = load_weights(root, manifest, read_json(inputs['pieceIndex']['path']))
    layers = []
    def progress(row):
        layers.append(row)
        print(json.dumps(row), flush=True)
    first_cache = {}
    def observe_cache(index, values):
        if index == config['layer_types'].index('full_attention'):
            first_cache.update(values)
    logits, caches = evaluate(contract['inputIds'], weights, config,
                             np.array(contract['prefixLengths']) - 1, progress,
                             observe_cache if boundary_capture else None)
    assert np.isfinite(logits).all()
    # Observations are loaded only after independent computation has completed.
    observed = read_json(inputs['capture']['path'])
    historical = read_json(inputs['historicalReference']['path'])
    sources = {key: read_json(inputs[key]['path']) for key in ('sourceChunk', 'sourceRecurrent') if key in inputs}
    results = []
    request = contract['requestIndex']
    for step, actual in zip(contract['steps'], logits, strict=True):
        target = next(row for row in observed['results'] if row['index'] == request and row['step'] == step)
        comparators = {'doppler': target['logits'], 'historical': historical['expected'][request]['steps'][step]['logits']}
        for key, source in sources.items():
            comparators[key] = next(row['logits'] for row in source['results'] if row['index'] == request and row['step'] == step)
        results.append({'index': request, 'step': step, 'greedyToken': int(np.argmax(actual)),
                        'logitsF64': base64.b64encode(actual.astype('<f8').tobytes()).decode(),
                        'comparisons': {key: comparison(actual, encoded, contract['tolerance']) for key, encoded in comparators.items()}})
        print(json.dumps({key: value for key, value in results[-1].items() if key != 'logitsF64'}), flush=True)
    report = {'schema': 'doppler.qwen35-independent-f64-diagnostic/v1', 'qualified': False,
              'scope': contract['scope'], 'contractSha256': digest(Path(contract_path).read_bytes()),
              'toolSha256': digest(Path(__file__).read_bytes()), 'tolerance': contract['tolerance'],
              'versions': {key: importlib.metadata.version(key) for key in ('numpy', 'gguf')},
              'shards': identities, 'layers': layers, 'cacheIdentities': caches, 'results': results}
    if boundary_capture:
        capture = read_json(boundary_capture)
        assert capture['manifestIdentity'] == 'sha256:' + inputs['manifest']['sha256']
        target = next(row for row in capture['results'] if row['index'] == request and row['step'] == contract['steps'][-1])
        assert target['logits'] == next(row['logits'] for row in observed['results'] if row['index'] == request and row['step'] == target['step'])
        report['firstCacheComparison'] = compare_first_cache(first_cache, capture, config)
        report['boundaryCaptureSha256'] = digest(Path(boundary_capture).read_bytes())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--boundary-capture', help='Optional retained GPU observations, compared only after independent evaluation')
    arguments = parser.parse_args()
    run(arguments.contract, arguments.out, arguments.boundary_capture)
