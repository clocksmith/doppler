#!/usr/bin/env python3
"""Matched intermediate comparisons; records error without changing acceptance gates."""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(policy):
    torch.set_num_threads(policy['threads'])
    quantized = np.load(policy['quantizedBoundaries'])
    reference = json.loads(Path(policy['referencePath']).read_text())
    for entry in reference['source']['files']:
        assert 'sha256:' + digest(Path(policy['sourceDir']) / entry['path']) == entry['hash']
    model = AutoModel.from_pretrained(policy['sourceDir'], local_files_only=True,
                                     dtype=torch.float32, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(policy['sourceDir'], local_files_only=True)
    source = {}
    handles = []
    for name, module in model.named_modules():
        if name in quantized:
            def hook(_module, _inputs, output, key=name):
                value = output[0] if isinstance(output, tuple) else output
                source[key] = value.detach().float().numpy().copy()
            handles.append(module.register_forward_hook(hook))
    encoded = tokenizer(reference['input']['texts'][0], return_tensors='pt', truncation=False)
    assert encoded.input_ids[0].tolist() == reference['outputs'][0]['tokenIds']
    with torch.inference_mode():
        model(**encoded, use_cache=False)
    for handle in handles:
        handle.remove()
    if Path(policy['sourceBoundaries']).exists():
        prior = np.load(policy['sourceBoundaries'])
        assert set(prior) == set(source) and all(np.array_equal(prior[key], value) for key, value in source.items())
    else:
        with Path(policy['sourceBoundaries']).open('xb') as stream:
            np.savez_compressed(stream, **source)
    mapping = {'embed.out': 'embed_tokens', 'layer.0.attn.post_input_norm': 'layers.0.input_layernorm',
               'layer.0.attn.q_proj': 'layers.0.self_attn.q_proj', 'layer.0.attn.k_proj': 'layers.0.self_attn.k_proj',
               'layer.0.attn.v_proj': 'layers.0.self_attn.v_proj', 'layer.0.layer.out': 'layers.0',
               'layer.27.layer.out': 'layers.27'}
    rows = []
    for mode in ['quantized', 'f16']:
        receipt = json.loads(Path(policy[mode + 'Doppler']).read_text())
        output = receipt['raw']['outputs'][0]
        assert output['tokenIds'] == reference['outputs'][0]['tokenIds']
        for op, name in mapping.items():
            capture = next(row['capture'] for row in output['diagnostics']['timeline'] if row['opId'] == op)
            ref = (quantized if mode == 'quantized' else source)[name].reshape(capture['shape'])
            expected = ref.reshape(-1) if 'data' in capture else np.asarray([ref[tuple(i)] for i in capture['sampleCoordinates']])
            actual = np.asarray(capture.get('data', capture['sample'])).reshape(-1)
            assert actual.shape == expected.shape and np.isfinite(actual).all()
            error = actual - expected
            rows.append({'mode': mode, 'operator': op, 'values': actual.size,
                         'maxAbsoluteError': float(np.max(np.abs(error))),
                         'rootMeanSquareError': float(np.sqrt(np.mean(error ** 2))),
                         'referenceMaxAbsolute': float(np.max(np.abs(expected)))})
    approximation = [{'boundary': name, 'maxAbsoluteError': float(np.max(abs(quantized[name] - source[name]))),
                      'rmsError': float(np.sqrt(np.mean((quantized[name] - source[name]) ** 2)))} for name in source]
    result = {'schema': 'doppler.quantized-boundary-comparison/v1', 'policy': policy,
              'probeSha256': digest(__file__),
              'inputs': {key: digest(policy[key]) for key in ['quantizedBoundaries', 'sourceBoundaries',
                         'referencePath', 'quantizedDoppler', 'f16Doppler']},
              'equivalentComputations': rows, 'quantizationApproximation': approximation,
              'scope': 'One input, complete selected matrices and layers, embedding slice. No new acceptance tolerance.'}
    with Path(policy['outputPath']).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    compare(json.loads(Path(sys.argv[1]).read_text()))
