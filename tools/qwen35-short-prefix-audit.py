#!/usr/bin/env python3
"""Observe the approved evaluator without editing it; audit a frozen request prefix.

CPU diagnostic only. Captured GPU operands enter local operation comparisons
after independent evaluation, never the authoritative reference state.
"""
import argparse
import base64
import gzip
import importlib.util
import inspect
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


def cache_double_rounding(values):
    """Diagnostic only: ideal F32 materialization before the unchanged F16 store."""
    direct = values.astype(np.float16).astype(np.float64)
    rounded = values.astype(np.float32).astype(np.float64)
    staged = rounded.astype(np.float16).astype(np.float64)
    coordinates = np.argwhere(direct != staged)
    return {'count': len(coordinates), 'values': int(values.size), 'coordinates': [
        {'index': list(map(int, c)), 'f64': float(values[tuple(c)]),
         'nearestF32': float(rounded[tuple(c)]), 'directF16': float(direct[tuple(c)]),
         'throughF32F16': float(staged[tuple(c)])} for c in coordinates]}


def compare_error_sources(actual, controlled, expected):
    """Compare signed errors, not merely similar maxima; diagnostic only."""
    assert actual.shape == controlled.shape == expected.shape
    left, right = actual - expected, controlled - expected
    a, b = left.reshape(-1), right.reshape(-1)
    actual_tail, control_tail = np.abs(left) > .001, np.abs(right) > .001
    overlap = actual_tail & control_tail
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    return {'actualAgainstControlled': error(actual, controlled),
        'signedErrorCosine': float(np.dot(a, b) / denominator) if denominator else None,
        'signedErrorCorrelation': correlation,
        'actualTail': int(actual_tail.sum()), 'controlTail': int(control_tail.sum()),
        'tailOverlap': int(overlap.sum()),
        'sameSignTailOverlap': int(np.count_nonzero(overlap & (np.signbit(left) == np.signbit(right)))),
        'perPrefix': [{'step': step, 'actualWorstVocabulary': int(np.argmax(np.abs(left[step]))),
            'controlWorstVocabulary': int(np.argmax(np.abs(right[step]))),
            'actualErrorAtActualWorst': float(left[step, np.argmax(np.abs(left[step]))]),
            'controlErrorAtActualWorst': float(right[step, np.argmax(np.abs(left[step]))]),
            'actualAgainstControlled': error(actual[step:step + 1], controlled[step:step + 1])}
            for step in range(len(actual))]}


def validate_runtime_context(runtime, binding, contract, history, history_hash):
    """History supplies forced tokens only, never authoritative accuracy values."""
    assert runtime['archiveSha256'] == binding['archiveSha256']
    assert runtime['manifestIdentity'] == 'sha256:' + binding['manifestSha256']
    assert runtime['referenceSha256'] == history_hash
    assert runtime['frozenManifestIdentity'] == history['modelIdentity']
    assert runtime['fullPrefixes'] is True
    assert runtime['physicalExecutionCompleted'] and not runtime['declaredShaderMismatches']
    rows = [row for row in runtime['results'] if row['index'] == contract['requestIndex']]
    assert [row['step'] for row in rows] == contract['steps']
    prefill = contract['prefixLengths'][0]
    assert all(row['inputIds'] == contract['inputIds'][:prefill] for row in rows)
    forced = [row['tokenId'] for row in history['expected'][contract['requestIndex']]['steps']]
    assert rows[0]['inputIds'] + forced[:len(rows) - 1] == contract['inputIds']
    return rows


def read_capture(capture, prefill=19, decode_steps=1):
    """Pair by actual generation step and operation; reject conflicting duplicates."""
    groups, step = {}, -1
    for row in capture['observation']['timeline']:
        if row['opId'] == 'embed.out':
            step += 1
        item = row.get('capture')
        if not item or item.get('level') != 'full':
            continue
        assert item['encoding'] == 'base64-f32le' and 0 <= step <= decode_steps
        value = np.frombuffer(base64.b64decode(item['dataBase64']), dtype='<f4')
        value = value.reshape(item['shape']).astype(np.float64)
        key = (row['opId'], step)
        if key in groups:
            assert np.array_equal(groups[key], value), key
        groups[key] = value
    assert step == decode_steps, 'Incomplete captured generation history'
    result = {}
    for name in sorted({name for name, _ in groups}):
        # Fused decode does not expose the prefill-only FFN gate/up tensors.
        # Required paired operations are accessed explicitly by the audit below.
        if any((name, index) not in groups for index in range(decode_steps + 1)):
            assert name in ('layer.0.ffn.gate', 'layer.0.ffn.up'), name
            continue
        parts = [groups[name, index] for index in range(decode_steps + 1)]
        assert parts[0].shape[0] == prefill, name
        assert all(part.shape[0] == 1 for part in parts[1:]), name
        result[name] = np.concatenate(parts)
    return result


def ideal_global_materialization(ref, weights, config, contract, expected, map_path,
                                 archive_sha256, capture_sha256):
    """Diagnostic copies only; ideal arithmetic between reviewed global stores."""
    policy_bytes = Path(map_path).read_bytes()
    policy = json.loads(policy_bytes)
    assert policy['reviewStatus'] == 'reviewed'
    assert policy['evaluatorSha256'] == EVALUATOR_SHA256
    assert policy['parentArchiveSha256'] == archive_sha256
    dispatch_bytes = Path(policy['dispatchBinding']['path']).read_bytes()
    assert ref.digest(dispatch_bytes) == policy['dispatchBinding']['sha256']
    dispatch = json.loads(dispatch_bytes)
    assert dispatch['archiveSha256'] == archive_sha256
    assert dispatch['captureSha256'] == capture_sha256
    assert policy['request'] == {'index': 2, 'prefillTokens': 184, 'lastDecodeStep': 7}
    assert contract['requestIndex'] == 2 and contract['prefixLengths'][0] == 184
    prefill, count = 184, 8
    enabled = False
    counts = {}

    def materialize(value, label):
        if not enabled:
            return value
        counts[label] = counts.get(label, 0) + int(np.size(value))
        return np.asarray(value).astype(np.float32).astype(np.float64)

    def pair_store(value):
        if not enabled:
            return value
        high = materialize(value, 'recurrent-high')
        low = materialize(value - high, 'recurrent-low')
        return high + low

    def prefill_store(value, label):
        if not enabled:
            return value
        result = value.copy()
        result[:prefill] = materialize(result[:prefill], label)
        return result

    def normalization(value, weight, epsilon, offset=True):
        epsilon = float(np.float32(epsilon)) if enabled else epsilon
        return materialize(ref.rms_norm(value, weight, epsilon, offset), 'normalization')

    def rotation(value, theta, dimension):
        if not enabled:
            return ref.text_rope(value, theta, dimension)
        frequencies = float(np.float32(theta)) ** (-np.arange(0, dimension, 2, dtype=np.float64) / dimension)
        angle = np.arange(len(value), dtype=np.float64)[:, None, None] * frequencies[None, None, :]
        cosine = materialize(np.cos(angle), 'rope-cos-table')
        sine = materialize(np.sin(angle), 'rope-sin-table')
        first, second = np.split(value[..., :dimension], 2, axis=-1)
        return materialize(np.concatenate((first * cosine - second * sine,
            second * cosine + first * sine, value[..., dimension:]), axis=-1), 'rope-output')

    delta_source = inspect.getsource(ref.delta_history)
    delta_changes = [
        ('        mixed *= sigmoid(mixed)',
         "        mixed *= sigmoid(mixed)\n        if index < prefill: mixed = materialize(mixed, 'conv-output')"),
        ('        state *= decay[:, None, None]',
         '        state *= decay[:, None, None]\n        state = pair_store(state)'),
        ('        state += key[:, :, None] * delta[:, None, :]',
         '        state += key[:, :, None] * delta[:, None, :]\n        state = pair_store(state)'),
        ('        output = np.sum(state * query[:, :, None], axis=1)',
         "        output = np.sum(state * query[:, :, None], axis=1)\n        output_unrounded = output\n        if index < prefill: output = materialize(output, 'raw-prefill-recurrent-output')"),
        ("        output = rms_norm(output, norm_weight, config['rms_norm_eps'], offset=False)",
         "        output = output / np.sqrt(np.mean(output_unrounded * output_unrounded, axis=-1, keepdims=True) + (float(np.float32(config['rms_norm_eps'])) if enabled_flag() else config['rms_norm_eps'])) * norm_weight"),
        ('        outputs.append((output * gate * sigmoid(gate)).reshape(-1))',
         "        outputs.append(materialize(output * gate * sigmoid(gate), 'gated-recurrent-output').reshape(-1))"),
        (' + 1e-6)', ' + (float(np.float32(1e-6)) if enabled_flag() else 1e-6))'),
    ]
    for before, after in delta_changes:
        assert delta_source.count(before) == (2 if before == ' + 1e-6)' else 1), before
        delta_source = delta_source.replace(before, after)
    delta_namespace = dict(ref.__dict__, materialize=materialize, pair_store=pair_store,
                           prefill=prefill, enabled_flag=lambda: enabled)
    exec(compile(delta_source, '<ideal-global-state-diagnostic>', 'exec'), delta_namespace)
    source = inspect.getsource(ref.evaluate)
    changes = [
        ("weights['embed_tokens.weight'][token_ids].copy()", "materialize(weights['embed_tokens.weight'][token_ids].copy(), 'embedding')"),
        ("normalized @ local('in_proj_qkv.weight').T", "materialize(normalized @ local('in_proj_qkv.weight').T, 'qkv-projection')"),
        ("normalized @ local('in_proj_z.weight').T", "materialize(normalized @ local('in_proj_z.weight').T, 'z-projection')"),
        ("normalized @ local('in_proj_a.weight').T", "materialize(normalized @ local('in_proj_a.weight').T, 'a-projection')"),
        ("normalized @ local('in_proj_b.weight').T", "materialize(normalized @ local('in_proj_b.weight').T, 'b-projection')"),
        ("core @ local('out_proj.weight').T", "materialize(core @ local('out_proj.weight').T, 'linear-output-projection')"),
        ("normalized @ local('q_proj.weight').T", "materialize(normalized @ local('q_proj.weight').T, 'q-projection')"),
        ("normalized @ local('k_proj.weight').T", "materialize(normalized @ local('k_proj.weight').T, 'k-projection')"),
        ("normalized @ local('v_proj.weight').T", "materialize(normalized @ local('v_proj.weight').T, 'v-projection')"),
        ('causal_attention(query, key, value) * sigmoid(gate)', "materialize(materialize(causal_attention(query, key, value), 'attention-output') * sigmoid(gate), 'gated-attention-output')"),
        ("core.reshape(len(x), -1) @ local('o_proj.weight').T", "materialize(core.reshape(len(x), -1) @ local('o_proj.weight').T, 'attention-output-projection')"),
        ('        x += update', "        norm_input = x + update\n        x = materialize(norm_input, 'attention-residual')"),
        ("normalized = rms_norm(x, w('post_attention_layernorm.weight'), epsilon)",
         "normalized = rms_norm(norm_input, w('post_attention_layernorm.weight'), epsilon)"),
        ("normalized @ w('mlp.gate_proj.weight').T", "prefill_store(normalized @ w('mlp.gate_proj.weight').T, 'ffn-gate-projection')"),
        ("normalized @ w('mlp.up_proj.weight').T", "prefill_store(normalized @ w('mlp.up_proj.weight').T, 'ffn-up-projection')"),
        ("        x += (gate * sigmoid(gate) * up) @ w('mlp.down_proj.weight').T", "        x = materialize(x + materialize(materialize(gate * sigmoid(gate) * up, 'ffn-activation') @ w('mlp.down_proj.weight').T, 'ffn-down-projection'), 'ffn-residual')"),
        ("normalized @ weights['lm_head.weight'].T", "materialize(normalized @ weights['lm_head.weight'].T, 'logits')"),
    ]
    for before, after in changes:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(ref.__dict__, materialize=materialize, prefill_store=prefill_store,
                     rms_norm=normalization, text_rope=rotation,
                     delta_history=delta_namespace['delta_history'])
    exec(compile(source, '<ideal-global-buffer-diagnostic>', 'exec'), namespace)
    evaluate = namespace['evaluate']
    baseline, _ = evaluate(contract['inputIds'], weights, config, np.array(contract['prefixLengths']) - 1)
    assert np.array_equal(baseline, expected), 'Inactive diagnostic copy must reproduce every approved vector'
    enabled = True
    logits, caches = evaluate(contract['inputIds'][:prefill + count - 1], weights, config,
                             np.array(contract['prefixLengths'][:count]) - 1)
    return {'scope': policy['scope'], 'mapSha256': ref.digest(policy_bytes),
        'evaluateSourceSha256': ref.digest(source.encode()), 'deltaSourceSha256': ref.digest(delta_source.encode()),
        'inactiveCopyApprovedVectorsExact': True, 'count': count, 'stores': counts,
        'comparison': error(logits, expected[:count]), 'caches': caches,
        'perPrefix': [{'step': i, 'comparison': error(logits[i:i + 1], expected[i:i + 1]),
            'logitsF64': base64.b64encode(logits[i].astype('<f8').tobytes()).decode()} for i in range(count)]}


def run(args):
    source = Path(__file__).with_name('qwen35-f64-reference.py')
    spec = importlib.util.spec_from_file_location('approved_f64', source)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    assert ref.digest(source.read_bytes()) == EVALUATOR_SHA256
    contract = ref.read_json(args.contract)
    request_index = contract['requestIndex']
    prefill = contract['prefixLengths'][0]
    capture = ref.read_json(args.capture)
    target = capture['observationTarget']
    assert target['index'] == request_index and target['physicalCache'] is True
    steps = target['step'] + 1
    assert 0 < steps <= len(contract['prefixLengths'])
    observed_tokens = prefill + steps - 1
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
            else:
                retain(prefix + 'core_out', ref.causal_attention(local['query'], local['key'], local['value']))
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
    approved_paths = [args.approved] if args.approved else [
        row['path'] for row in ref.read_json(args.approved_index)['shards']
        if row['index'] == request_index]
    approved_rows = []
    for path in approved_paths:
        approved = ref.read_json(path)
        assert approved['toolSha256'] == EVALUATOR_SHA256
        assert approved['contractSha256'] == ref.digest(Path(args.contract).read_bytes())
        approved_rows.extend(approved['results'])
    assert [row['step'] for row in approved_rows] == contract['steps']
    expected = np.stack([np.frombuffer(base64.b64decode(row['logitsF64']), dtype='<f8')
                         for row in approved_rows])
    assert np.array_equal(logits, expected), 'Observation must exactly reproduce approved vectors'
    approved_cache_targets = [ref.half_cache(independent[f'layer.{layer}.attn.{name}']).reshape(
        len(contract['inputIds']), config['num_key_value_heads'], config['head_dim'])
        for layer, kind in enumerate(config['layer_types']) if kind == 'full_attention'
        for name in ('k_rope', 'v_proj')]
    for name in independent:
        independent[name] = independent[name][:steps if name == 'final_norm.out' else observed_tokens]
    logits = logits[:steps]
    assert capture['observation']['maximumUnmaskedDifference'] == 0
    manifest_identity = contract['inputs']['manifest']['sha256']
    if args.binding:
        binding = ref.read_json(args.binding)
        assert binding['sourceManifestSha256'] == manifest_identity
        assert capture['archiveSha256'] == binding['archiveSha256']
        manifest_identity = binding['manifestSha256']
    assert capture['manifestIdentity'] == 'sha256:' + manifest_identity
    assert capture['declaredShaderMismatches'] == []
    gpu = read_capture(capture, prefill, steps - 1)
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
            actual_input = gpu[f'layer.{layer}.attn.post_input_norm'][:prefill]
            actual_output = gpu[f'layer.{layer}.attn.{operation}'][:prefill]
            input_bytes = actual_input.astype('<f4').tobytes()
            encode = lambda value: base64.b64encode(value).decode()
            fixture = {'M': prefill, 'N': entry['shape'][0], 'K': entry['shape'][1],
                'operation': f'layer.{layer}.attn.{operation}', 'input': encode(input_bytes),
                'inputSha256': ref.digest(input_bytes), 'weights': encode(raw), 'weightsSha256': ref.digest(raw),
                'oracleFloat64': encode((actual_input @ weights[name].T).astype('<f8').tobytes()),
                'baseline026': encode(actual_output.astype('<f4').tobytes()),
                'captureSha256': ref.digest(Path(args.capture).read_bytes())}
            (directory / f'layer-{layer}-{operation}.json').write_text(json.dumps(fixture) + '\n')
        float_cases = []
        for layer in (0, 1):
            for suffix in ('a', 'b'):
                name = f'layers.{layer}.linear_attn.in_proj_{suffix}.weight'
                operation = f'layer.{layer}.attn.linear_{suffix}_proj'
                actual_input = gpu[f'layer.{layer}.attn.post_input_norm'][:prefill]
                weight = weights[name]
                assert np.array_equal(weight, weight.astype('<f2').astype(np.float64))
                encode = lambda value: base64.b64encode(value).decode()
                float_cases.append({'name': operation, 'M': prefill, 'N': weight.shape[0], 'K': weight.shape[1],
                    'inputF32': encode(actual_input.astype('<f4').tobytes()),
                    'weightF16': encode(weight.astype('<f2').tobytes()),
                    'expectedF64': encode((actual_input @ weight.T).astype('<f8').tobytes()),
                    'observedF32': encode(gpu[operation][:prefill].astype('<f4').tobytes())})
        (directory / 'linear-ab-f16.json').write_text(json.dumps({
            'schema': 'doppler.f16-projection-actual-operands/v1',
            'captureSha256': ref.digest(Path(args.capture).read_bytes()), 'cases': float_cases}) + '\n')
        layer = 15
        prefix = f'layer.{layer}.attn.'
        nh, nk, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
        query = gpu[prefix + 'q_rope'][:prefill].reshape(-1, nh, width)
        key = ref.half_cache(gpu[prefix + 'k_rope'][:prefill].reshape(-1, nk, width))
        value = ref.half_cache(gpu[prefix + 'v_proj'][:prefill].reshape(-1, nk, width))
        attention_expected = ref.causal_attention(query, key, value)
        attention_fixture = {'scope': 'Same captured operands; independent local attention counterfactual only',
            'captureSha256': ref.digest(Path(args.capture).read_bytes()), 'evaluatorSha256': EVALUATOR_SHA256,
            'layer': layer, 'headDim': width, 'numHeads': nh, 'numKVHeads': nk, 'seqLen': prefill,
            'startPos': 0, 'isCausal': True, 'scale': width ** -0.5,
            'q': query.reshape(-1).tolist(), 'k': key.reshape(-1).tolist(), 'v': value.reshape(-1).tolist(),
            'float64Expected': attention_expected.reshape(-1).tolist(),
            'observedF32': gpu[prefix + 'core_out'][:prefill].reshape(-1).tolist()}
        (directory / 'attention-layer15.json.gz').write_bytes(gzip.compress(
            json.dumps(attention_fixture).encode(), mtime=0))
    rows, storage = [], []
    replay_comparisons = []
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
            if args.normalization_replay and layer == 3:
                fixture = ref.read_json(args.normalization_fixture)
                assert fixture['captureSha256'] == ref.digest(Path(args.capture).read_bytes())
                assert fixture['operation'] == prefix + 'k_norm'
                assert fixture['headDim'] == width
                assert np.array_equal(np.frombuffer(base64.b64decode(fixture['kWeight']), dtype='<f2').astype(np.float64),
                                      w('self_attn.k_norm.weight'))
                assert np.float32(fixture['epsilon']) == np.float32(config['rms_norm_eps'])
                assert np.array_equal(np.asarray(fixture['k']).reshape(key.shape),
                                      gpu[prefix + 'k_proj'].reshape(key.shape))
                actual_norm = gpu[prefix + 'k_norm']
                ideal_norm = independent[prefix + 'k_norm']
                local_norm = key.reshape(actual_norm.shape)
                for replay_path in args.normalization_replay:
                    replay = ref.read_json(replay_path)
                    assert replay['fixtureSha256'] == ref.digest(Path(args.normalization_fixture).read_bytes())
                    assert replay['passed'] and not replay['errors']
                    values = np.asarray(replay['cases'][0]['values'], dtype=np.float64)
                    assert len(values) == 2 * actual_norm.size
                    candidate = values[actual_norm.size:].reshape(actual_norm.shape)
                    controls = {'captured': actual_norm, 'candidate': candidate,
                                'sameInputF64': local_norm}
                    result = {'replaySha256': ref.digest(Path(replay_path).read_bytes()),
                              'shaderSha256': replay['shaderSha256'],
                              'local': error(candidate, local_norm),
                              'total': error(candidate, ideal_norm), 'stored': {}}
                    rotary = config['rope_parameters']
                    for scope in ('nonrotary', 'allWithIdealRoPE'):
                        start = int(width * rotary['partial_rotary_factor'])
                        transform = (lambda value: value.reshape(-1, nk, width)[:, :, start:]) if scope == 'nonrotary' else (
                            lambda value: ref.text_rope(value.reshape(-1, nk, width),
                                rotary['rope_theta'], start))
                        target_half = ref.half_cache(transform(ideal_norm))
                        variants = {name: ref.half_cache(transform(value)) for name, value in controls.items()}
                        old, new, local_half = (variants[name] for name in ('captured', 'candidate', 'sameInputF64'))
                        result['stored'][scope] = {
                            'scope': 'Same captured inputs; unchanged independent target; no downstream state replay',
                            'capturedDifferences': int(np.count_nonzero(old != target_half)),
                            'candidateDifferences': int(np.count_nonzero(new != target_half)),
                            'sameInputF64Differences': int(np.count_nonzero(local_half != target_half)),
                            'corrected': int(np.count_nonzero((old != target_half) & (new == target_half))),
                            'newDifferences': int(np.count_nonzero((old == target_half) & (new != target_half))),
                            'removedCancellationOfInheritedError': int(np.count_nonzero(
                                (old == target_half) & (new != target_half) & (local_half != target_half)))}
                    replay_comparisons.append(result)
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
                    'idealF32DoubleRounding': cache_double_rounding(b),
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
    gpu_norm = gpu['final_norm.out'][np.array(contract['prefixLengths'][:steps]) - 1]
    rows.append({'operation': 'final_norm.out', 'total': error(gpu_norm, independent['final_norm.out']),
                 'local': error(gpu_norm, ref.rms_norm(gpu['layer.23.layer.out'][np.array(contract['prefixLengths'][:steps]) - 1],
                                                     weights['norm.weight'], config['rms_norm_eps']))})
    actual_logits = np.stack([np.frombuffer(base64.b64decode(row['logits']), dtype='<f4').astype(np.float64)
                              for row in capture['results']])
    local_logits = gpu_norm @ weights['lm_head.weight'].T
    worst_vocabulary = int(np.argmax(np.abs(actual_logits[-1] - logits[-1])))
    report = {'schema': 'doppler.short-prefix-f64-audit/v1', 'qualified': False,
        'scope': 'Independent-state total error versus same-GPU-operands local error; diagnostic only',
        'evaluatorSha256': EVALUATOR_SHA256, 'auditSha256': ref.digest(Path(__file__).read_bytes()),
        'captureSha256': ref.digest(Path(args.capture).read_bytes()),
        'approvedVectors': [{'path': path, 'sha256': ref.digest(Path(path).read_bytes())} for path in approved_paths],
        'approvedVectorsExact': True, 'shards': identities, 'cacheIdentities': caches,
        'operations': rows, 'storage': storage, 'logits': {
            'total': error(actual_logits, logits), 'local': error(actual_logits, local_logits),
            'inherited': error(local_logits, logits), 'selectedPrefixWorst': {
                'index': request_index, 'step': steps - 1, 'vocabulary': worst_vocabulary,
                'gpu': float(actual_logits[-1, worst_vocabulary]),
                'f64': float(logits[-1, worst_vocabulary]),
                'localF64': float(local_logits[-1, worst_vocabulary])}}}
    if replay_comparisons:
        report['normalizationReplay'] = replay_comparisons
    if args.ideal_materialization_map:
        report['idealGlobalMaterializationControl'] = ideal_global_materialization(
            ref, weights, config, contract, expected, args.ideal_materialization_map,
            capture['archiveSha256'], ref.digest(Path(args.capture).read_bytes()))
    if args.cache_materialization_control:
        # Counterfactual diagnostic, not an acceptance reference. All other
        # equations stay F64; only the mandatory producer F32 store is inserted.
        original_half_cache = ref.half_cache
        try:
            cache_call = 0
            def staged_half_cache(value):
                nonlocal cache_call
                kind = 'key' if cache_call % 2 == 0 else 'value'
                cache_call += 1
                selected = args.cache_materialization_kind in ('both', kind)
                return original_half_cache(value.astype(np.float32).astype(np.float64) if selected else value)
            ref.half_cache = staged_half_cache
            staged_logits, _ = ref.evaluate(contract['inputIds'], weights, config,
                np.array(contract['prefixLengths']) - 1)
        finally:
            ref.half_cache = original_half_cache
        per_step = np.max(np.abs(staged_logits - expected), axis=1)
        report['cacheMaterializationControl'] = {
            'scope': 'F64 equations with F32 producer materialization before F16 cache; diagnostic only',
            'referenceUnchanged': True, 'materializedKind': args.cache_materialization_kind,
            'comparison': error(staged_logits, expected),
            'prefixesOverTolerance': int(np.count_nonzero(per_step > 0.001)),
            'perPrefixMaximum': list(map(float, per_step))}
    if args.recurrent_materialization_control:
        # Diagnostic lower-level storage intervention. The authoritative source
        # and independently evolved reference above remain unchanged.
        original_delta = ref.delta_history
        original_source = inspect.getsource(original_delta)
        boundary = '        state += key[:, :, None] * delta[:, None, :]'
        assert original_source.count(boundary) == 1
        store = '\n        state = state.astype(np.float32).astype(np.float64)'
        if args.recurrent_storage_precision == 'two-f32':
            store = ('\n        state_high = state.astype(np.float32).astype(np.float64)'
                     '\n        state_low = (state - state_high).astype(np.float32).astype(np.float64)'
                     '\n        state = state_high + state_low')
        controlled_source = original_source.replace(boundary, boundary + store)
        namespace = dict(ref.__dict__)
        exec(compile(controlled_source, '<diagnostic-f32-recurrent-storage>', 'exec'), namespace)
        original_independent = independent
        original_half_cache = ref.half_cache
        cache_calls = 0
        def frozen_approved_cache(value):
            nonlocal cache_calls
            target = approved_cache_targets[cache_calls]
            cache_calls += 1
            assert value.shape == target.shape
            return target.copy()
        if args.recurrent_attribution_step is not None:
            assert 0 <= args.recurrent_attribution_step < steps
            independent = {}
            assert sys.gettrace() is None
            sys.settrace(trace)
        try:
            ref.delta_history = namespace['delta_history']
            if args.recurrent_approved_cache_control:
                ref.half_cache = frozen_approved_cache
            staged_logits, staged_caches = ref.evaluate(contract['inputIds'], weights, config,
                np.array(contract['prefixLengths']) - 1,
                observe_cache=observe_cache if args.recurrent_attribution_step is not None else None)
        finally:
            if args.recurrent_attribution_step is not None:
                sys.settrace(None)
            staged_independent = independent
            independent = original_independent
            ref.delta_history = original_delta
            ref.half_cache = original_half_cache
        if args.recurrent_approved_cache_control:
            assert cache_calls == len(approved_cache_targets)
        per_step = np.max(np.abs(staged_logits - expected), axis=1)
        report['recurrentMaterializationControl'] = {
            'scope': 'F64 equations with declared recurrent-state storage after each update; diagnostic only',
            'storagePrecision': args.recurrent_storage_precision,
            'cacheIdentities': staged_caches,
            'cacheIdentitiesMatchApproved': staged_caches == caches,
            'referenceUnchanged': True, 'controlledSourceSha256': ref.digest(controlled_source.encode()),
            'approvedCacheIntervention': args.recurrent_approved_cache_control,
            'cacheInterventionScope': 'Diagnostic fixed independently evolved F16 K/V; inadmissible for qualification'
                if args.recurrent_approved_cache_control else None,
            'comparison': error(staged_logits, expected),
            'prefixesOverTolerance': int(np.count_nonzero(per_step > 0.001)),
            'perPrefixMaximum': list(map(float, per_step))}
        if args.recurrent_attribution_step is not None:
            position = prefill + args.recurrent_attribution_step - 1
            attribution = []
            for layer, kind in enumerate(config['layer_types']):
                prefix = f'layer.{layer}.attn.'
                w = lambda name: weights[f'layers.{layer}.' + name]
                before = gpu['embed.out' if layer == 0 else f'layer.{layer - 1}.layer.out']
                local_norm = ref.rms_norm(before, w('input_layernorm.weight'), config['rms_norm_eps'])
                checkpoints = [('post_input_norm', local_norm)]
                if kind == 'linear_attention':
                    local_core = namespace['delta_history'](*[gpu[prefix + name] for name in
                        ('qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj')],
                        w('linear_attn.conv1d.weight').squeeze(1), w('linear_attn.A_log'),
                        w('linear_attn.dt_bias'), w('linear_attn.norm.weight'), config)
                    checkpoints.append(('linear_core_out', local_core))
                else:
                    nh, nk, width = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
                    local_core = ref.causal_attention(gpu[prefix + 'q_rope'].reshape(-1, nh, width),
                        ref.half_cache(gpu[prefix + 'k_rope'].reshape(-1, nk, width)),
                        ref.half_cache(gpu[prefix + 'v_proj'].reshape(-1, nk, width)))
                    checkpoints.append(('core_out', local_core.reshape(len(before), -1)))
                for suffix, local in checkpoints:
                    name = prefix + suffix
                    actual, controlled = gpu[name][position:position + 1], staged_independent[name][position:position + 1]
                    introduced = local[position:position + 1]
                    attribution.append({'operation': name, 'actualVsStorageControl': error(actual, controlled),
                        'sameInputLocal': error(actual, introduced), 'inherited': error(introduced, controlled)})
                name = f'layer.{layer}.layer.out'
                attribution.append({'operation': name, 'actualVsStorageControl': error(
                    gpu[name][position:position + 1], staged_independent[name][position:position + 1])})
                if kind == 'full_attention':
                    for suffix in ('k_rope', 'v_proj'):
                        name = prefix + suffix
                        actual = ref.half_cache(gpu[name][:position + 1])
                        controlled = ref.half_cache(staged_independent[name][:position + 1])
                        differences = np.argwhere(actual != controlled)
                        attribution.append({'operation': name + '.stored', 'differentValues': len(differences),
                            'firstDifference': list(map(int, differences[0])) if len(differences) else None,
                            'actualVsStorageControl': error(actual, controlled)})
            report['recurrentMaterializationControl']['selectedPrefixAttribution'] = {
                'step': args.recurrent_attribution_step, 'tokenPosition': position, 'operations': attribution}
        if args.recurrent_runtime:
            runtime = ref.read_json(args.recurrent_runtime)
            binding = ref.read_json(args.binding)
            historical = contract['inputs']['historicalReference']
            history_bytes = Path(historical['path']).read_bytes()
            assert ref.digest(history_bytes) == historical['sha256']
            history_bytes = gzip.decompress(history_bytes)
            runtime_rows = validate_runtime_context(runtime, binding, contract,
                json.loads(history_bytes), ref.digest(history_bytes))
            actual = np.stack([np.frombuffer(base64.b64decode(row['logits']), dtype='<f4').astype(np.float64)
                               for row in runtime_rows])
            report['recurrentMaterializationControl']['runtimeComparison'] = {
                'path': args.recurrent_runtime, 'sha256': ref.digest(Path(args.recurrent_runtime).read_bytes()),
                'archiveSha256': runtime['archiveSha256'],
                **compare_error_sources(actual, staged_logits, expected)}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--approved')
    parser.add_argument('--approved-index', default='artifacts/whole-model-f64-complete-20261011/independent-output-index.json')
    parser.add_argument('--contract', default='artifacts/whole-model-f64-complete-20261011/request-0-contract.json')
    parser.add_argument('--binding', help='Exact archive/source manifest binding when the runtime shader identities changed')
    parser.add_argument('--out', required=True)
    parser.add_argument('--recurrent-materialization-control', action='store_true')
    parser.add_argument('--recurrent-storage-precision', choices=['f32', 'two-f32'], default='f32',
                        help='Independent diagnostic storage only; never changes runtime precision')
    parser.add_argument('--recurrent-runtime', help='Same-archive full-prefix output for signed-error attribution')
    parser.add_argument('--recurrent-attribution-step', type=int, help='Observe storage-control intermediates at one retained prefix')
    parser.add_argument('--recurrent-approved-cache-control', action='store_true',
                        help='Diagnostic only: replace K/V cache with independently evolved approved values')
    parser.add_argument('--cache-materialization-control', action='store_true')
    parser.add_argument('--ideal-materialization-map', help='Reviewed global-buffer materialization diagnostic map')
    parser.add_argument('--cache-materialization-kind', choices=['both', 'key', 'value'], default='both')
    parser.add_argument('--fixture-directory', help='Export captured prefill projections for the existing physical replay')
    parser.add_argument('--normalization-fixture', help='Exact captured layer-3 fixture for local K comparisons')
    parser.add_argument('--normalization-replay', action='append', help='Physical same-input Q/K receipt; diagnostic only')
    args = parser.parse_args()
    if (args.recurrent_runtime or args.recurrent_attribution_step is not None
        or args.recurrent_approved_cache_control or args.recurrent_storage_precision != 'f32'):
        if not args.recurrent_materialization_control:
            parser.error('Recurrent diagnostic options require --recurrent-materialization-control')
    run(args)
