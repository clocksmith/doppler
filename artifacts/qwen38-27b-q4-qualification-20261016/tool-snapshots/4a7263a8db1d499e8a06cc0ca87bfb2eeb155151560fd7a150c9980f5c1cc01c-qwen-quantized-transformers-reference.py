#!/usr/bin/env python3
"""Independent Transformers CPU control over exact retained Q4 tensors, streamed by layer."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

spec = importlib.util.spec_from_file_location('stream_reference', Path(__file__).with_name('qwen-quantized-stream-reference.py'))
stream_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream_reference)


class StoredHalfCache(implementation.DynamicCache):
    def update(self, keys, values, layer_idx, *args, **kwargs):
        keys, values = super().update(keys.half(), values.half(), layer_idx, *args, **kwargs)
        return keys.float(), values.float()


def run(contract_path, model_root, output):
    torch.set_num_threads(4)
    contract = json.loads(Path(contract_path).read_text())
    for entry in contract['inputs'].values():
        assert stream_reference.file_hash(entry['path']) == entry['sha256']
    expected_source = json.loads(Path(contract['inputs']['transcript']['path']).read_text())
    assert importlib.metadata.version('transformers') == expected_source['referenceImplementation']['version']
    assert 'sha256:' + stream_reference.file_hash(implementation.__file__) == expected_source['referenceImplementation']['modelingSourceDigest']
    inventory = json.loads(Path(contract['inputs']['inventory']['path']).read_text())
    root = Path(model_root)
    assert stream_reference.file_hash(root / 'manifest.json') == inventory['manifestSha256']
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['artifactIdentity']['sourceRevision'] == contract['sourceRevision']
    source = json.loads(Path(contract['inputs']['sourceConfig']['path']).read_text())['text_config']
    config = Qwen3_5TextConfig(**source)
    config.tie_word_embeddings = False
    config._attn_implementation = 'eager'
    config.dtype = 'float32'
    with torch.device('meta'):
        model = implementation.Qwen3_5ForCausalLM(config)
    model.model.rotary_emb = implementation.Qwen3_5TextRotaryEmbedding(config=config, device='cpu')
    model.eval()
    weights = stream_reference.RetainedWeights(root, manifest, inventory, dtype=np.float32)
    handles = []
    modules = [('embed_tokens', model.model.embed_tokens), ('norm', model.model.norm), ('lm_head', model.lm_head)]
    modules += [(f'layers.{i}', layer) for i, layer in enumerate(model.model.layers)]
    for prefix, module in modules:
        def prepare(current, _args, prefix=prefix):
            values = {name: torch.from_numpy(weights[prefix + '.' + name]) for name in current.state_dict()}
            current.load_state_dict(values, strict=True, assign=True)
        def release(current, _args, _output, prefix=prefix):
            current.to_empty(device='meta')
            if prefix.startswith('layers.'):
                print(json.dumps({'stage': 'layer', 'layer': prefix}), flush=True)
        handles.append(module.register_forward_pre_hook(prepare))
        handles.append(module.register_forward_hook(release))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    states = []
    try:
        for case in contract['cases']:
            start = time.monotonic()
            cache = StoredHalfCache(config=config)
            with torch.inference_mode():
                hidden = model.model(input_ids=torch.tensor([case['inputIds']], dtype=torch.long),
                                     past_key_values=cache, use_cache=True).last_hidden_state
                logits = model.lm_head(hidden[:, case['positions']]).squeeze(0).numpy().copy()
            for layer in contract.get('stateLayers', []):
                for field, attribute in [('recurrentState', 'recurrent_states'), ('convState', 'conv_states')]:
                    values = getattr(cache.layers[layer], attribute).float().numpy().copy()
                    assert np.isfinite(values).all()
                    state_file = output / f'{case["id"]}-layer-{layer}-{field}.npy'
                    np.save(state_file, values, allow_pickle=False)
                    states.append(dict(caseId=case['id'], layer=layer, field=field, shape=list(values.shape),
                                       position=len(case['inputIds']) - 1, path=state_file.name,
                                       sha256=stream_reference.file_hash(state_file)))
            assert np.isfinite(logits).all()
            filename = output / (case['id'] + '.npy')
            np.save(filename, logits, allow_pickle=False)
            rows.append(dict(id=case['id'], positions=case['positions'], greedyTokens=np.argmax(logits, axis=1).tolist(),
                             logits=dict(path=filename.name, sha256=stream_reference.file_hash(filename), shape=list(logits.shape)),
                             elapsedSeconds=time.monotonic() - start))
    finally:
        for handle in handles:
            handle.remove()
    result = dict(schema='doppler.qwen-quantized-transformers-reference/v1', qualified=False,
                  scope='Independent Transformers equations and gguf decoder; F32 arithmetic, F16 KV storage',
                  versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'gguf', 'numpy']},
                  contractSha256=stream_reference.file_hash(contract_path), toolSha256=stream_reference.file_hash(__file__),
                  manifestSha256=inventory['manifestSha256'], cases=rows, states=states,
                  maxRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    (output / 'reference.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.contract, args.model, args.out)
