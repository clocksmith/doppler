#!/usr/bin/env python3
"""Offline original-weight control for an already captured Qwen3.5 generation.

Only weights and their original tying differ from the deployed-weight control.
Prompt IDs, greedy policy, repetition penalty, F32 source equations and F16 KV
storage remain fixed. This is attribution evidence, never runtime qualification.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_tensor_name(name):
    if not name.startswith('model.language_model.'):
        return None
    return name.replace('model.language_model.', 'model.', 1)


def run(args):
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer

    reference_path = Path(__file__).with_name('qwen35-deployed-reference.py')
    spec = importlib.util.spec_from_file_location('deployed_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(args.threads)
    root = Path(args.source)
    receipt = reference.read_json(root / 'download-receipt.json')
    assert receipt['repository'] == 'Qwen/Qwen3.5-2B'
    assert receipt['revision'] == args.revision and len(args.revision) == 40
    for entry in receipt['files']:
        filename = root / entry['path']
        assert filename.parent == root and filename.stat().st_size == entry['sizeBytes']
        assert file_sha256(filename) == entry['sha256'], entry['path']
        assert f'/resolve/{args.revision}/' in entry['url']

    control = reference.read_json(args.control)
    deployed = Path(args.deployed)
    manifest = reference.read_json(deployed / 'manifest.json')
    assert control['modelIdentity'] == 'sha256:' + file_sha256(deployed / 'manifest.json')
    assert manifest['artifactIdentity']['sourceCheckpointId'] == receipt['repository']
    assert manifest['inference']['session']['compute']['defaults']['activationDtype'] == 'f32'
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    original_tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    deployed_tokenizer = Tokenizer.from_file(str(deployed / 'tokenizer.json'))
    assert original_tokenizer.get_vocab() == deployed_tokenizer.get_vocab()
    prompt_ids = control['result']['tokenIds']
    prompt = deployed_tokenizer.decode(prompt_ids, skip_special_tokens=False)
    assert original_tokenizer.encode(prompt, add_special_tokens=False).ids == prompt_ids
    source = reference.read_json(root / 'config.json')['text_config']
    assert source['hidden_size'] == manifest['architecture']['hiddenSize']
    assert source['layer_types'] == manifest['inference']['layerPattern']['layerTypes']
    assert source['rms_norm_eps'] == manifest['inference']['normalization']['rmsNormEps']
    assert source['tie_word_embeddings'] is True
    config = reference.Qwen3_5TextConfig(**source)
    config.dtype = 'float32'
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = reference.Qwen3_5ForCausalLM(config)
    model.model.rotary_emb = reference.Qwen3_5TextRotaryEmbedding(config, device='cpu')
    wanted = model.state_dict()
    index = reference.read_json(root / 'model.safetensors.index.json')['weight_map']
    weights = {}
    source_dtypes = set()
    for filename in sorted(set(index.values())):
        assert any(entry['path'] == filename for entry in receipt['files'])
        with safe_open(root / filename, framework='pt', device='cpu') as checkpoint:
            for name in checkpoint.keys():
                key = source_tensor_name(name)
                if key is None:
                    continue
                assert index[name] == filename and key in wanted, name
                tensor = checkpoint.get_tensor(name)
                assert tensor.shape == wanted[key].shape
                source_dtypes.add(str(tensor.dtype))
                weights[key] = tensor.float()
    assert 'lm_head.weight' not in weights
    weights['lm_head.weight'] = weights['model.embed_tokens.weight']
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    model.eval()
    del weights
    report = {
        'schema': 'doppler.original-weight-generation-diagnostic/v1', 'qualified': False,
        'scope': 'Original checkpoint weights with deployed generation policy and F16 KV storage',
        'modelIdentity': control['modelIdentity'], 'source': receipt,
        'sourceConfigSha256': file_sha256(root / 'config.json'),
        'originalTokenizerSha256': file_sha256(root / 'tokenizer.json'),
        'tokenizerSha256': file_sha256(deployed / 'tokenizer.json'),
        'toolSha256': file_sha256(__file__), 'generationToolSha256': file_sha256(reference_path),
        'versions': {name: reference.importlib.metadata.version(name)
                     for name in ['torch', 'transformers', 'safetensors', 'tokenizers']},
        'precision': {'arithmetic': 'float32', 'kvStorage': 'float16',
                      'weights': sorted(source_dtypes), 'originalTiedHead': True},
        'threads': args.threads,
    }
    replay = SimpleNamespace(generation_control=args.control, model=args.deployed, out=args.out)
    report = reference.replay_generation(model, config, manifest, replay, report)
    report['schema'] = 'doppler.original-weight-generation-diagnostic/v1'
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['source', 'revision', 'deployed', 'control', 'out']:
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--threads', type=int, required=True)
    arguments = parser.parse_args()
    assert arguments.threads > 0
    run(arguments)
