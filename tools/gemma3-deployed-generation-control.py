#!/usr/bin/env python3
"""Offline Gemma 3 control: source equations, identified deployed weights/inputs.

This diagnoses one retained application failure; it neither qualifies a model
nor changes a runtime, manifest, numerical fixture or application policy.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from transformers.initialization import no_init_weights
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

_spec = importlib.util.spec_from_file_location('deployed_reference', Path(__file__).with_name('qwen35-deployed-reference.py'))
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
read_json, sha256, replay_generation = _reference.read_json, _reference.sha256, _reference.replay_generation


def decode_weight(data, entry):
    shape = entry['shape']
    if len(data) != entry['size']:
        raise ValueError('Stored tensor byte length differs')
    if entry['dtype'] == 'Q4_K_M':
        if len(shape) != 2 or entry['layout'] != 'row':
            raise ValueError('Expected row-padded Q4_K matrix')
        rows, columns = shape
        padded = ((columns + 255) // 256) * 256
        if len(data) != rows * padded // 256 * 144:
            raise ValueError('Q4_K row storage differs from shape')
        array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
        array = array.reshape(rows, padded)[:, :columns].copy()
    elif entry['dtype'] == 'BF16':
        array = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    elif entry['dtype'] in ('F16', 'F32'):
        array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4').astype(np.float32)
    else:
        raise ValueError('Unsupported deployed encoding: ' + entry['dtype'])
    return torch.from_numpy(array.reshape(shape).copy())


def run(args):
    torch.set_num_threads(args.threads)
    root = Path(args.model)
    manifest = read_json(root / 'manifest.json')
    binding = read_json(args.binding)
    control = read_json(args.generation_control) if args.generation_control else None
    identity = 'sha256:' + sha256(root / 'manifest.json')
    assert identity == binding['preparedManifest'] == args.model_identity
    if control:
        assert control['modelIdentity'] == identity
    source = read_json(args.source_config)
    assert source['model_type'] == 'gemma3_text'
    for source_key, deployed_key in [('hidden_size', 'hiddenSize'), ('num_hidden_layers', 'numLayers'),
                                     ('num_attention_heads', 'numAttentionHeads'), ('num_key_value_heads', 'numKeyValueHeads'),
                                     ('head_dim', 'headDim'), ('vocab_size', 'vocabSize')]:
        assert source[source_key] == manifest['architecture'][deployed_key]
    assert source['rms_norm_eps'] == manifest['inference']['normalization']['rmsNormEps']
    assert manifest['inference']['output']['tieWordEmbeddings'] is True
    assert manifest['inference']['session']['compute']['defaults']['activationDtype'] == 'f32'
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    config = Gemma3TextConfig(**source)
    config.dtype = 'float32'
    config._attn_implementation = 'eager'
    # Keep source-owned rotary/embedding buffers initialized on CPU; parameter
    # random initialization is unnecessary because every parameter is replaced.
    with no_init_weights():
        model = Gemma3ForCausalLM(config)
    files = {item['file']: item['sha256'] for item in binding['files']}
    assert sha256(root / 'tokenizer.json') == files['tokenizer.json']
    shards = {}
    for index, shard in enumerate(manifest['shards']):
        path = root / shard['filename']
        assert sha256(path) == files[shard['filename']]
        shards[index] = path.read_bytes()
        assert len(shards[index]) == shard['size']
    weights = {}
    encoded_w8a8 = {}
    for name, entry in manifest['tensors'].items():
        spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
        data = b''.join(shards[span['shardIndex']][span['offset']:span['offset'] + span['size']] for span in spans)
        if entry['dtype'] == 'W8A8':
            from lib.w8a8_model_reference import decode_rows
            codes, scales = decode_rows(data, entry)
            encoded_w8a8[name] = (codes, scales)
            # Strict state loading still checks every named tensor/shape. These
            # placeholder weights are replaced by integer-operation modules.
            weights[name] = torch.from_numpy(codes.astype(np.float32) * scales[:, None])
        else:
            weights[name] = decode_weight(data, entry)
    weights['lm_head.weight'] = weights['model.embed_tokens.weight']
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval()
    if encoded_w8a8:
        from lib.w8a8_model_reference import install_w8a8_modules
        installed_w8a8 = install_w8a8_modules(model, encoded_w8a8)
    else:
        installed_w8a8 = {}
    del shards, weights
    report = {'schema': 'doppler.source-generation-diagnostic/v1', 'qualified': False,
              'scope': 'Independent source generation using deployed Gemma weights; no numerical or distributed qualification',
              'modelIdentity': identity, 'sourceConfigSha256': sha256(args.source_config),
              'sourceRevision': args.source_revision, 'toolSha256': sha256(__file__),
              'sharedReplaySha256': sha256(Path(__file__).with_name('qwen35-deployed-reference.py')),
              'bindingSha256': sha256(args.binding), 'files': binding['files'],
              'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'gguf', 'numpy']},
              'precision': {'arithmetic': 'float32', 'kvStorage': 'float16', 'weights': 'identified deployed bytes'}}
    if args.quality_suite:
        from lib.w8a8_model_quality import evaluate_deployed_quality
        report = evaluate_deployed_quality(model, config, manifest, args, report)
    else:
        report['expectedOutputText'] = control['result']['evidence']['outputText']
        report = replay_generation(model, config, manifest, args, report)
        report['sameTokens'] = report['generatedTokenIds'] == control['result']['evidence']['tokenIds']
    if installed_w8a8:
        report['w8a8Reference'] = {
            'implementationSha256': sha256(Path(__file__).parent / 'lib/w8a8_model_reference.py'),
            'scope': 'Exact integer-code GEMM via binary64 with ordered F32 rescale and explicit FTZ',
            'modules': {name: {'calls': module.calls, 'rows': module.rows} for name, module in installed_w8a8.items()},
        }
        assert all(module.calls > 0 for module in installed_w8a8.values())
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'model-identity', 'binding', 'source-config', 'source-revision', 'out'):
        parser.add_argument('--' + name, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generation-control')
    mode.add_argument('--quality-suite')
    parser.add_argument('--source-tokenizer')
    parser.add_argument('--threads', type=int, default=8)
    run(parser.parse_args())
