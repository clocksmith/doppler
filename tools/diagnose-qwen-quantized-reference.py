#!/usr/bin/env python3
"""Independent CPU oracle for exact retained Q4_K bytes; never a Run fallback."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors import safe_open


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(policy):
    torch.set_num_threads(policy['threads'])
    root = Path(policy['candidateDir'])
    manifest = json.loads((root / 'manifest.json').read_text())
    receipt = json.loads(Path(policy['candidateReceipt']).read_text())
    assert digest(root / 'manifest.json') == receipt['manifestSha256']
    for shard in receipt['shards']:
        filename = root / shard['file']
        assert filename.stat().st_size == shard['bytes'] and digest(filename) == shard['sha256']
    reference = json.loads(Path(policy['referencePath']).read_text())
    source = Path(policy['sourceDir'])
    for entry in reference['source']['files']:
        assert 'sha256:' + digest(source / entry['path']) == entry['hash']
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = AutoModel.from_config(config, attn_implementation='eager')
    # RoPE frequencies are non-persistent buffers, absent from state_dict.
    model.rotary_emb = type(model.rotary_emb)(config=config, device='cpu')
    weights = {}
    samples = []
    for name, entry in manifest['tensors'].items():
        spans = entry.get('spans', [{'shardIndex': entry.get('shard'), 'offset': entry.get('offset'), 'size': entry['size']}])
        chunks = []
        for span in spans:
            with (root / manifest['shards'][span['shardIndex']]['filename']).open('rb') as stream:
                stream.seek(span['offset']); chunks.append(stream.read(span['size']))
        data = b''.join(chunks)
        assert len(data) == entry['size']
        if entry['dtype'] == 'Q4_K_M':
            assert entry['layout'] == 'row'
            array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
        else:
            assert entry['dtype'] in ('F16', 'F32')
            array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4').astype(np.float32)
        array = array.reshape(entry['shape'])
        weights[name] = torch.from_numpy(array.copy())
        if name.startswith('layers.0.') and entry['dtype'] == 'Q4_K_M':
            samples.append({'name': name, 'packedSha256': hashlib.sha256(data).hexdigest(),
                            'decodedSha256': hashlib.sha256(array.tobytes()).hexdigest(),
                            'firstBlock': array.reshape(-1)[:256].tolist()})
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    observed = json.loads(Path(policy['dopplerPath']).read_text())['raw']['outputs']
    boundaries = {}
    handles = []
    for name, module in model.named_modules():
        if name in policy['boundaries']:
            def hook(_module, _inputs, output, key=name):
                value = output[0] if isinstance(output, tuple) else output
                boundaries[key] = value.detach().float().numpy().copy()
            handles.append(module.register_forward_hook(hook))
    outputs = []
    for index, target in enumerate(reference['outputs']):
        encoded = tokenizer(target['text'], return_tensors='pt', truncation=False)
        assert encoded.input_ids[0].tolist() == target['tokenIds'] == observed[index]['tokenIds']
        with torch.inference_mode():
            hidden = model(**encoded, use_cache=False).last_hidden_state
            vector = torch.nn.functional.normalize(hidden[:, -1], p=2, dim=1)[0].numpy()
        expected = np.asarray(target['embedding'], dtype=np.float32)
        actual = np.asarray(observed[index]['embedding'], dtype=np.float32)
        outputs.append({'text': target['text'], 'tokenIds': target['tokenIds'], 'embedding': vector.tolist(),
                        'quantizedVsOriginalMaxError': float(np.max(np.abs(vector - expected))),
                        'dopplerVsQuantizedMaxError': float(np.max(np.abs(vector - actual)))})
        if index == 0:
            with Path(policy['boundaryPath']).open('xb') as stream:
                np.savez_compressed(stream, **boundaries)
        print(index, outputs[-1]['quantizedVsOriginalMaxError'], outputs[-1]['dopplerVsQuantizedMaxError'], flush=True)
    for handle in handles:
        handle.remove()
    sensitivity = []
    # Change one weight group at a time; always restore the exact decoded baseline.
    with safe_open(source / 'model.safetensors', framework='pt') as original:
        for group in policy.get('restoreGroups', []):
            names = [name for name, entry in manifest['tensors'].items()
                     if entry['dtype'] == 'Q4_K_M' and group in name]
            assert names, 'Empty restoration group'
            saved = {name: model.get_parameter(name).detach().clone() for name in names}
            with torch.no_grad():
                for name in names:
                    model.get_parameter(name).copy_(original.get_tensor(name).half().float())
            errors = []
            for target in reference['outputs']:
                encoded = tokenizer(target['text'], return_tensors='pt', truncation=False)
                with torch.inference_mode():
                    hidden = model(**encoded, use_cache=False).last_hidden_state[:, -1]
                    vector = torch.nn.functional.normalize(hidden, p=2, dim=1)[0].numpy()
                errors.append(float(np.max(np.abs(vector - np.asarray(target['embedding'])))))
            added = sum(int(np.prod(manifest['tensors'][name]['shape'])) * 2 - manifest['tensors'][name]['size'] for name in names)
            sensitivity.append({'group': group, 'tensorCount': len(names), 'additionalWeightBytes': added,
                                'maximumError': max(errors), 'errors': errors,
                                'scope': 'CPU sensitivity, not a converted or qualified mixed-precision release'})
            with torch.no_grad():
                for name in names:
                    model.get_parameter(name).copy_(saved[name])
            print('Restored group', group, max(errors), added, flush=True)
    result = {'schema': 'doppler.exact-quantized-reference/v1', 'policy': policy,
              'decoder': {'package': 'gguf', 'version': importlib.metadata.version('gguf')},
              'torchVersion': torch.__version__, 'dtype': 'float32', 'attention': 'eager',
              'manifestSha256': receipt['manifestSha256'], 'referenceSha256': digest(policy['referencePath']),
              'dopplerSha256': digest(policy['dopplerPath']), 'probeSha256': digest(__file__),
              'boundarySha256': digest(policy['boundaryPath']), 'weightSamples': samples, 'outputs': outputs,
              'sensitivity': sensitivity,
              'signedCapsuleExecution': False, 'releaseQualified': False}
    with Path(policy['outputPath']).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
