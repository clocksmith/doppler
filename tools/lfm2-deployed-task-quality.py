#!/usr/bin/env python3
"""Offline fresh-task screen of existing LFM2 weights, never a Run fallback."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from lib.source_task_quality import evaluate_tasks

import blake3
import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2RotaryEmbedding


def digest(data):
    return hashlib.sha256(data).hexdigest()



def run(args):
    torch.set_num_threads(args.threads)
    root, source = Path(args.model), Path(args.source)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    suite_bytes = Path(args.suite).read_bytes()
    suite = json.loads(suite_bytes)
    assert manifest['modelId'] == suite['modelId']
    source_receipt = json.loads((source / 'download-receipt.json').read_text())
    for entry in source_receipt['files']:
        data = (source / entry['path']).read_bytes()
        assert len(data) == entry['bytes'] and digest(data) == entry['sha256']
        assert f"/resolve/{source_receipt['revision']}/" in entry['url']
    config = Lfm2Config(**json.loads((source / 'config.json').read_text()))
    config.dtype, config._attn_implementation = 'float32', 'eager'
    assert config.layer_types == manifest['inference']['layerPattern']['layerTypes']
    assert config.hidden_size == manifest['architecture']['hiddenSize']
    assert config.num_hidden_layers == manifest['architecture']['numLayers']
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    with torch.device('meta'):
        model = Lfm2ForCausalLM(config)
    model.model.rotary_emb = Lfm2RotaryEmbedding(config, device='cpu')
    wanted, weights, shards, identities = model.state_dict(), {}, {}, []
    for index, shard in enumerate(manifest['shards']):
        data = (root / shard['filename']).read_bytes()
        assert len(data) == shard['size']
        shards[index] = data
        identities.append({'file': shard['filename'], 'bytes': len(data), 'sha256': digest(data),
                           'legacyManifestHash': shard['hash'], 'standardBlake3': blake3.blake3(data).hexdigest()})
    for name, entry in manifest['tensors'].items():
        assert name in wanted, name
        assert list(wanted[name].shape) == entry['shape'], name
        spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
        data = b''.join(shards[x['shardIndex']][x['offset']:x['offset'] + x['size']] for x in spans)
        assert len(data) == entry['size']
        if entry['dtype'] == 'Q4_K_M':
            array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
        elif entry['dtype'] == 'BF16':
            array = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
        else:
            assert entry['dtype'] in ('F16', 'F32')
            array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4').astype(np.float32)
        weights[name] = torch.from_numpy(array.copy().reshape(entry['shape']))
    assert config.tie_word_embeddings
    weights['lm_head.weight'] = weights['model.embed_tokens.weight']
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    model.eval()
    del weights, shards
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    source_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    report = {'schema': 'doppler.source-task-quality-result/v1', 'qualified': False,
              'modelIdentity': 'sha256:' + digest(manifest_bytes), 'suiteSha256': digest(suite_bytes),
              'producerSha256': digest(Path(__file__).read_bytes()), 'source': source_receipt,
              'evaluatorSha256': digest((Path(__file__).parent / 'lib/source_task_quality.py').read_bytes()),
              'weights': identities, 'packages': {name: importlib.metadata.version(name)
                  for name in ['torch', 'transformers', 'gguf', 'tokenizers']},
              'scope': suite['scope'], 'arithmetic': 'F32 independent source equations, F16 stored attention K/V',
              'settings': suite['generation'], 'results': []}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report) + '\n')
    evaluate_tasks(model, config, tokenizer, source_tokenizer, suite, report, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['model', 'source', 'suite', 'out']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--threads', type=int, required=True)
    run(parser.parse_args())
