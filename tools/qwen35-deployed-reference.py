#!/usr/bin/env python3
"""Independent, offline Transformers control using identified deployed RDRR weights.

This diagnostic never changes Run or activates an acceptance reference. Q4_K
unpacking comes from gguf; model equations come from Transformers. The explicit
KV boundary matches the deployed F16 storage with F32 attention arithmetic.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import blake3
import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding, torch_recurrent_gated_delta_rule,
)


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StoredHalfCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states.half(), value_states.half(), layer_idx, *args, **kwargs)
        return keys.float(), values.float()


def compare(actual, encoded):
    expected = np.frombuffer(base64.b64decode(encoded), dtype='<f4')
    assert expected.shape == actual.shape and np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    error = np.abs(actual - expected)
    return {'maxAbsError': float(error.max()), 'rmsError': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
            'sampledToken': int(actual.argmax()), 'comparedValues': int(actual.size),
            'withinTolerance': bool(error.max() <= 0.001)}


def capture_values(capture):
    if not capture:
        return None
    if capture.get('data') is not None:
        return np.asarray(capture['data'], dtype=np.float32)
    if capture.get('dataBase64') is not None:
        assert capture.get('encoding') == 'base64-f32le'
        return np.frombuffer(base64.b64decode(capture['dataBase64'], validate=True), dtype='<f4')
    return None


def compare_attention_history(model, capture, cache):
    """Compare observed attention against its actual captured cache operands.

    Reconstruct only history emitted by this capture. Missing tokens or shapes
    reject the diagnostic rather than substituting independently evolved state.
    No reconstructed value is fed back into either execution.
    """
    timeline = capture['observation']['timeline']
    layer_indices = sorted({int(row['opId'].split('.')[1]) for row in timeline
                            if row['opId'].startswith('layer.')
                            and row['opId'].endswith('.attn.core_out')
                            and capture_values(row.get('capture')) is not None})
    comparisons = []
    for index in layer_indices:
        attention = model.model.layers[index].self_attn
        head_dim = attention.head_dim
        kv_heads = model.config.num_key_value_heads
        heads = model.config.num_attention_heads
        prefix = f'layer.{index}.attn.'
        histories = {'k_rope': [], 'v_proj': []}
        last = {}
        for row in timeline:
            if row['opId'] == 'embed.out':
                last = {}
            if not row['opId'].startswith(prefix):
                continue
            data = capture_values(row.get('capture'))
            if data is None:
                continue
            name = row['opId'][len(prefix):]
            values = np.asarray(data, dtype=np.float32)
            assert np.isfinite(values).all(), row['opId']
            last[name] = values
            if name in histories:
                histories[name].append(values.reshape(-1, kv_heads, head_dim))
        query = last['q_rope'].reshape(-1, heads, head_dim)
        # This boundary diagnostic intentionally requires a single decode token.
        assert query.shape[0] == 1, 'attention history requires a decode capture'
        actual = last['core_out'].reshape(heads, head_dim).astype(np.float64)
        stored = {name: np.concatenate(parts).astype(np.float16)
                  for name, parts in histories.items()}
        reference_layer = cache.layers[index]
        cache_rows = []
        for name, source in [('k_rope', reference_layer.keys), ('v_proj', reference_layer.values)]:
            source = source.detach().cpu().numpy()[0].transpose(1, 0, 2)
            observed = stored[name]
            assert source.dtype == observed.dtype == np.float16
            assert source.shape == observed.shape, 'capture must include complete cache history'
            error = np.abs(source.astype(np.float64) - observed.astype(np.float64))
            cache_rows.append({'boundary': prefix + name, 'tokens': int(source.shape[0]),
                               'values': int(source.size),
                               'differentStoredValues': int(np.count_nonzero(source != observed)),
                               'maxAbsError': float(error.max()),
                               'rmsError': float(np.sqrt(np.mean(error ** 2))),
                               'sourceCacheSha256': hashlib.sha256(source.copy().tobytes()).hexdigest(),
                               'capturedCacheSha256': hashlib.sha256(observed.tobytes()).hexdigest()})
        keys = np.repeat(stored['k_rope'].astype(np.float64), heads // kv_heads, axis=1)
        values = np.repeat(stored['v_proj'].astype(np.float64), heads // kv_heads, axis=1)
        scores = np.einsum('hd,thd->ht', query[0].astype(np.float64), keys) * attention.scaling
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        precise = np.einsum('ht,thd->hd', probabilities, values)
        error = np.abs(actual - precise)
        comparisons.append({'boundary': prefix + 'core_out',
                            'scope': 'Captured query and full captured K/V history, F16 storage, float64 reference',
                            'values': int(actual.size), 'tokens': int(keys.shape[0]),
                            'maxAbsError': float(error.max()),
                            'rmsError': float(np.sqrt(np.mean(error ** 2))),
                            'independentlyEvolvedCache': cache_rows})
    return comparisons


def compare_boundaries(model, capture, input_ids, cache):
    """Observe the selected prefix; hooks never replace model values."""
    retained = {}
    timeline = capture['observation']['timeline']
    start = max(index for index, row in enumerate(timeline) if row['opId'] == 'embed.out')
    for row in timeline[start:]:
        values = capture_values(row.get('capture'))
        if values is not None:
            retained[row['opId']] = {**row['capture'], 'data': values}
    modules = {'embed.out': model.model.embed_tokens, 'final_norm.out': model.model.norm}
    for index, layer in enumerate(model.model.layers):
        modules[f'layer.{index}.attn.post_input_norm'] = layer.input_layernorm
        modules[f'layer.{index}.layer.out'] = layer
    layer = model.model.layers[0]
    for name, module in {
        'attn.qkv_proj': layer.linear_attn.in_proj_qkv,
        'attn.linear_z_proj': layer.linear_attn.in_proj_z,
        'attn.linear_a_proj': layer.linear_attn.in_proj_a,
        'attn.linear_b_proj': layer.linear_attn.in_proj_b,
        'attn.out': layer.linear_attn.out_proj,
        'ffn.in': layer.post_attention_layernorm,
        'ffn.gate': layer.mlp.gate_proj,
        'ffn.up': layer.mlp.up_proj,
        'ffn.out': layer.mlp.down_proj,
    }.items():
        modules['layer.0.' + name] = module
    attention = model.model.layers[3].self_attn
    for name in ['q_proj', 'k_proj', 'v_proj', 'q_norm', 'k_norm', 'o_proj']:
        boundary = 'out' if name == 'o_proj' else name
        if 'layer.3.attn.' + boundary in retained:
            modules['layer.3.attn.' + boundary] = getattr(attention, name)
    comparisons, handles = [], []

    def observe(name):
        def hook(_module, _inputs, value):
            if isinstance(value, tuple):
                value = value[0]
            if name == 'layer.3.attn.q_proj':
                value = value.reshape(-1, model.config.num_attention_heads, attention.head_dim * 2)[..., :attention.head_dim]
            actual = value.detach().float().cpu().numpy().reshape(-1)
            expected = np.asarray(retained[name]['data'], dtype=np.float32)
            assert actual.shape == expected.shape, name
            error = np.abs(actual - expected)
            assert np.isfinite(error).all(), name
            comparisons.append({'boundary': name, 'values': int(actual.size),
                                'maxAbsError': float(error.max()),
                                'rmsError': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                                'actualSha256': hashlib.sha256(actual.tobytes()).hexdigest()})
        return hook

    operands = []
    def compare_operand(name, input_name, module, kind):
        x = np.asarray(retained[input_name]['data'], dtype=np.float32).reshape(-1, module.weight.shape[-1])
        expected = np.asarray(retained[name]['data'], dtype=np.float32).reshape(-1)
        native = module(torch.from_numpy(x.copy())).detach().float().numpy().reshape(-1)
        weight = module.weight.detach().float().numpy().astype(np.float64)
        precise_input = x.astype(np.float64)
        if kind == 'rmsnorm':
            precise = precise_input / np.sqrt(np.mean(precise_input ** 2, axis=-1, keepdims=True) + module.eps)
            precise *= (1 + weight)
        else:
            assert module.bias is None
            precise = precise_input @ weight.T
        precise = precise.reshape(-1)
        assert expected.shape == native.shape == precise.shape
        operands.append({'boundary': name, 'inputBoundary': input_name, 'operation': kind,
                         'values': int(expected.size),
                         'gpuVsFloat64': float(np.max(np.abs(expected.astype(np.float64) - precise))),
                         'sourceF32VsFloat64': float(np.max(np.abs(native.astype(np.float64) - precise))),
                         'gpuVsSourceF32SameOperands': float(np.max(np.abs(expected - native)))})

    try:
        for name, module in modules.items():
            if name not in retained:
                continue  # Fused decode kernels need not expose unfused FFN intermediates.
            handles.append(module.register_forward_hook(observe(name)))
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        assert len(comparisons) == len(handles)
        for handle in handles:
            handle.remove()
        handles.clear()
        for index, layer in enumerate(model.model.layers):
            compare_operand(f'layer.{index}.attn.post_input_norm',
                            'embed.out' if index == 0 else f'layer.{index - 1}.layer.out',
                            layer.input_layernorm, 'rmsnorm')
        compare_operand('final_norm.out', 'final_norm.pre', model.model.norm, 'rmsnorm')
        for name in ['qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj']:
            compare_operand('layer.0.attn.' + name, 'layer.0.attn.post_input_norm',
                            modules['layer.0.attn.' + name], 'linear')
        for name in ['gate', 'up']:
            if 'layer.0.ffn.' + name not in retained:
                continue
            compare_operand('layer.0.ffn.' + name, 'layer.0.ffn.in', modules['layer.0.ffn.' + name], 'linear')
        if 'layer.3.attn.k_rope' in retained:
            for name in ['k_proj', 'v_proj']:
                compare_operand('layer.3.attn.' + name, 'layer.3.attn.post_input_norm', getattr(attention, name), 'linear')
            compare_operand('layer.3.attn.q_norm', 'layer.3.attn.q_proj', attention.q_norm, 'rmsnorm')
            compare_operand('layer.3.attn.k_norm', 'layer.3.attn.k_proj', attention.k_norm, 'rmsnorm')
        history = compare_attention_history(model, capture, cache) if input_ids.shape[-1] == 1 else []
        return output, comparisons, operands, [name for name in modules if name not in retained], history
    finally:
        for handle in handles:
            handle.remove()


def replay_generation(model, config, manifest, args, report):
    """Independent greedy replay of an identified deployed generation receipt."""
    from tokenizers import Tokenizer
    control = read_json(args.generation_control)
    assert control['modelIdentity'] == report['modelIdentity']
    settings = control['input']['generation']
    assert settings['temperature'] == 0 and settings['topK'] == 1 and settings['topP'] == 1
    assert settings['presencePenalty'] == 0 and settings['stopSequences'] == []
    assert settings['suppressTokenIds'] == [] and settings['repetitionPenaltyWindow'] == 0
    tokenizer = Tokenizer.from_file(str(Path(args.model) / 'tokenizer.json'))
    prompt = control['result']['tokenIds']
    assert tokenizer.encode(tokenizer.decode(prompt, skip_special_tokens=False), add_special_tokens=False).ids == prompt
    expected = control['result']['evidence']['tokenIds']
    assert len(prompt) + settings['maxTokens'] <= settings['maxSeqLen']
    report.update({'schema': 'doppler.source-generation-diagnostic/v1',
                   'generationControlSha256': sha256(args.generation_control),
                   'tokenization': 'Exact captured IDs; independently round-tripped through deployed tokenizer',
                   'inputIds': prompt, 'settings': settings, 'eosTokenIds': manifest['eos_token_id'],
                   'generatedTokenIds': [], 'firstTokenDivergence': None, 'stopReason': None})
    cache = StoredHalfCache(config=config)
    history = list(prompt)
    input_ids = prompt
    for step in range(settings['maxTokens']):
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([input_ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits = result.logits[0, -1].float().numpy().copy()
        assert np.isfinite(logits).all()
        # Independent scalar expression of the declared full-history repetition penalty.
        for token in set(history):
            value = float(logits[token])
            logits[token] = value / settings['repetitionPenalty'] if value > 0 else value * settings['repetitionPenalty']
        if config.pad_token_id is not None:
            logits[config.pad_token_id] = -np.inf
        token = int(logits.argmax())
        report['generatedTokenIds'].append(token)
        if report['firstTokenDivergence'] is None and (step >= len(expected) or token != expected[step]):
            report['firstTokenDivergence'] = {'step': step, 'sourceToken': token,
                                              'dopplerToken': expected[step] if step < len(expected) else None}
        history.append(token)
        input_ids = [token]
        if token in manifest['eos_token_id']:
            report['stopReason'] = 'eos'
        elif step + 1 == settings['maxTokens']:
            report['stopReason'] = 'max-tokens'
        if step % 32 == 0 or report['stopReason']:
            report['outputText'] = tokenizer.decode(report['generatedTokenIds'], skip_special_tokens=True)
            Path(args.out).write_text(json.dumps(report) + '\n')
            print(json.dumps({'step': step, 'token': token, 'stopReason': report['stopReason'],
                              'firstTokenDivergence': report['firstTokenDivergence'],
                              'textTail': report['outputText'][-100:]}), flush=True)
        if report['stopReason']:
            break
    return report


def run(args):
    torch.set_num_threads(args.threads)
    root = Path(args.model)
    manifest = read_json(root / 'manifest.json')
    observed = read_json(args.capture)
    assert observed['manifestIdentity'] == 'sha256:' + sha256(root / 'manifest.json')
    assert manifest['inference']['session']['compute']['defaults']['activationDtype'] == 'f32'
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    source = read_json(args.source_config)['text_config']
    assert source['hidden_size'] == manifest['architecture']['hiddenSize']
    assert source['layer_types'] == manifest['inference']['layerPattern']['layerTypes']
    assert source['rms_norm_eps'] == manifest['inference']['normalization']['rmsNormEps']
    config = Qwen3_5TextConfig(**source)
    # The deployed LM head is separately quantized; retying would change weights.
    config.tie_word_embeddings = False
    config.dtype = 'float32'
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device='cpu')
    wanted = model.state_dict()
    shards, weights, identities = {}, {}, []
    pieces = read_json(args.piece_index)
    assert 'sha256:' + sha256(args.piece_index) == args.piece_index_identity
    files = {entry['path']: entry for entry in pieces['files']}
    tokenizer_bytes = (root / 'tokenizer.json').read_bytes()
    assert len(tokenizer_bytes) == files['tokenizer.json']['size']
    for piece in files['tokenizer.json']['pieces']:
        block = tokenizer_bytes[piece['offset']:piece['offset'] + piece['size']]
        assert 'sha256:' + hashlib.sha256(block).hexdigest() == piece['identity']
    assert manifest['hashAlgorithm'] == 'blake3'
    for index, shard in enumerate(manifest['shards']):
        filename = root / shard['filename']
        data = filename.read_bytes()
        assert len(data) == shard['size'] == files[shard['filename']]['size']
        cursor = 0
        for piece in files[shard['filename']]['pieces']:
            assert piece['offset'] == cursor
            block = data[cursor:cursor + piece['size']]
            assert 'sha256:' + hashlib.sha256(block).hexdigest() == piece['identity']
            cursor += piece['size']
        assert cursor == len(data)
        shards[index] = data
        identities.append({'filename': shard['filename'], 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                           'legacyManifestHash': shard['hash'], 'standardBlake3': blake3.blake3(data).hexdigest()})
    for name, entry in manifest['tensors'].items():
        if not name.startswith('model.language_model.'):
            continue
        key = name.replace('model.language_model.', 'model.', 1)
        if key == 'model.lm_head.weight':
            key = 'lm_head.weight'
        assert key in wanted, key
        spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
        data = b''.join(shards[x['shardIndex']][x['offset']:x['offset'] + x['size']] for x in spans)
        assert len(data) == entry['size']
        if entry['dtype'] == 'Q4_K_M':
            assert entry['layout'] == 'row'
            array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
        else:
            assert entry['dtype'] in ('F16', 'F32')
            array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4').astype(np.float32)
        weights[key] = torch.from_numpy(array.copy().reshape(wanted[key].shape))
    del shards
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.eval()
    if args.linear_prefill == 'source-recurrent':
        # Independent source sensitivity control, never a runtime substitution.
        # Both functions implement the same recurrence with different F32 order.
        for layer in model.model.layers:
            if hasattr(layer, 'linear_attn'):
                layer.linear_attn.chunk_gated_delta_rule = torch_recurrent_gated_delta_rule
    report = {'schema': 'doppler.source-model-reference-diagnostic/v1', 'qualified': False,
              'scope': 'Independent model-equation diagnostic, not reference activation or runtime fallback',
              'linearPrefillReference': args.linear_prefill,
              'tolerance': 0.001, 'tokenization': 'Exact captured prompt IDs, followed by frozen reference tokens',
              'modelIdentity': observed['manifestIdentity'], 'sourceConfigSha256': sha256(args.source_config),
              'captureSha256': sha256(args.capture), 'pieceIndexIdentity': args.piece_index_identity,
              'pieceIndexManifestIdentity': pieces['manifestIdentity'],
              'integrityScope': 'Pinned SHA-256 pieces, separate from the retained legacy shard hash implementation', 'toolSha256': sha256(__file__),
              'tokenizerSha256': sha256(root / 'tokenizer.json'), 'shards': identities,
              'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'gguf', 'numpy']},
              'precision': {'arithmetic': 'float32', 'kvStorage': 'float16', 'weights': 'exact deployed bytes decoded to float32'},
              'results': []}
    if args.generation_control:
        return replay_generation(model, config, manifest, args, report)
    reference = read_json(args.reference)
    reference_bytes = Path(args.reference).read_bytes()
    reference_bytes = gzip.decompress(reference_bytes) if reference_bytes[:2] == b'\x1f\x8b' else reference_bytes
    assert observed['referenceSha256'] == hashlib.sha256(reference_bytes).hexdigest()
    report['referenceSha256'] = observed['referenceSha256']
    targets = observed['results'][:args.prefixes]
    assert len(targets) == args.prefixes
    boundary_capture = read_json(args.boundary_capture) if args.boundary_capture else None
    if boundary_capture:
        target = targets[-1]
        bound = boundary_capture.get('observationTarget', {'index': 0, 'step': 0})
        assert (target['index'], target['step']) == (bound['index'], bound['step'])
        matching = [row for row in boundary_capture['results']
                    if row['index'] == target['index'] and row['step'] == target['step']][-1]
        assert matching['inputIds'] == target['inputIds']
        assert matching['logits'] == target['logits']
        report['boundaryCaptureSha256'] = sha256(args.boundary_capture)
        report['boundaryCaptureManifestIdentity'] = boundary_capture['manifestIdentity']
    cache, active_request, next_step = None, None, 0
    for target in targets:
        if target['step'] == 0:
            active_request, next_step = target['index'], 0
            cache = StoredHalfCache(config=config)
            input_ids = target['inputIds']
        else:
            input_ids = [reference['expected'][target['index']]['steps'][target['step'] - 1]['tokenId']]
        assert active_request == target['index'] and next_step == target['step']
        next_step += 1
        with torch.inference_mode():
            if boundary_capture and target is targets[-1]:
                result, report['boundaryComparisons'], report['operandComparisons'], report['unobservedBoundaries'], report['attentionHistoryComparisons'] = compare_boundaries(
                    model, boundary_capture, torch.tensor([input_ids]), cache)
            else:
                result = model(input_ids=torch.tensor([input_ids]), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
            output = result.logits[0, -1].float().numpy().copy()
        row = {'index': target['index'], 'step': target['step'], 'ordinal': target['ordinal'],
               'inputIds': input_ids, 'comparison': compare(output, target['logits']),
               'logitsSha256': hashlib.sha256(output.tobytes()).hexdigest(),
               'logits': base64.b64encode(output.astype('<f4').tobytes()).decode()}
        report['results'].append(row)
        Path(args.out).write_text(json.dumps(report) + '\n')
        print(json.dumps({key: value for key, value in row.items() if key != 'logits'}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['model', 'source-config', 'capture', 'out', 'reference', 'piece-index', 'piece-index-identity']:
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--prefixes', type=int, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--generation-control', help='Replay exact deployed greedy generation with independent source equations')
    parser.add_argument('--linear-prefill', choices=['source-chunk', 'source-recurrent'], default='source-chunk',
                        help='Independent source recurrence ordering sensitivity; never reference activation')
    parser.add_argument('--boundary-capture', help='Retained GPU observations for the last requested prefix')
    options = parser.parse_args()
    assert options.prefixes > 0 and options.threads > 0
    run(options)
