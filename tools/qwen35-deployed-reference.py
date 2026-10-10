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
    DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding,
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
    report = {'schema': 'doppler.source-model-reference-diagnostic/v1', 'qualified': False,
              'scope': 'Independent model-equation diagnostic, not reference activation or runtime fallback',
              'tolerance': 0.001, 'tokenization': 'Exact captured prompt IDs, followed by frozen reference tokens',
              'modelIdentity': observed['manifestIdentity'], 'sourceConfigSha256': sha256(args.source_config),
              'captureSha256': sha256(args.capture), 'pieceIndexIdentity': args.piece_index_identity,
              'pieceIndexManifestIdentity': pieces['manifestIdentity'],
              'integrityScope': 'Pinned SHA-256 pieces, separate from the retained legacy shard hash implementation', 'toolSha256': sha256(__file__),
              'tokenizerSha256': sha256(root / 'tokenizer.json'), 'shards': identities,
              'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'gguf', 'numpy']},
              'precision': {'arithmetic': 'float32', 'kvStorage': 'float16', 'weights': 'exact deployed bytes decoded to float32'},
              'results': []}
    reference = read_json(args.reference)
    reference_bytes = Path(args.reference).read_bytes()
    reference_bytes = gzip.decompress(reference_bytes) if reference_bytes[:2] == b'\x1f\x8b' else reference_bytes
    assert observed['referenceSha256'] == hashlib.sha256(reference_bytes).hexdigest()
    report['referenceSha256'] = observed['referenceSha256']
    targets = observed['results'][:args.prefixes]
    assert len(targets) == args.prefixes
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
            output = model(input_ids=torch.tensor([input_ids]), past_key_values=cache,
                           use_cache=True, logits_to_keep=1).logits[0, -1].float().numpy().copy()
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
    options = parser.parse_args()
    assert options.prefixes > 0 and options.threads > 0
    run(options)
