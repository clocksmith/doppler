#!/usr/bin/env python3
"""Offline exact-weight reference with bounded tensor residency, never a Run fallback."""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, dequantize


def import_tool(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tensor_chunks = import_tool('inventory_quantized_model', Path(__file__).with_name('inventory-quantized-model.py')).tensor_chunks


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RetainedWeights:
    def __init__(self, root, manifest, inventory, dtype=np.float64):
        self.root, self.manifest = root, manifest
        self.entries = {row['name']: row for row in inventory['tensors']}
        self.reads = 0
        self.dtype = dtype

    def __getitem__(self, name):
        source = name if name == 'lm_head.weight' else 'model.language_model.' + name
        entry = self.manifest['tensors'][source]
        frozen = self.entries[source]
        data = b''.join(tensor_chunks(self.root, self.manifest, entry))
        assert hashlib.sha256(data).hexdigest() == frozen['sha256'], source
        assert entry['dtype'] == frozen['dtype'] and entry['shape'] == frozen['shape']
        if entry['dtype'] == 'Q4_K_M':
            assert entry['layout'] == 'row'
            array = dequantize(np.frombuffer(data, dtype=np.uint8), GGMLQuantizationType.Q4_K)
        else:
            assert entry['dtype'] in ('F16', 'F32')
            array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4')
        self.reads += 1
        return array.reshape(entry['shape']).astype(self.dtype)


def run(contract_path, model_dir, output):
    contract = json.loads(Path(contract_path).read_text())
    assert contract['schema'] == 'doppler.qwen-quantized-reference-input/v1'
    for item in contract['inputs'].values():
        assert file_hash(item['path']) == item['sha256'], item['path']
    inventory = json.loads(Path(contract['inputs']['inventory']['path']).read_text())
    root = Path(model_dir)
    assert file_hash(root / 'manifest.json') == inventory['manifestSha256']
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['artifactIdentity']['sourceRevision'] == contract['sourceRevision']
    config = json.loads(Path(contract['inputs']['sourceConfig']['path']).read_text())['text_config']
    assert config['hidden_size'] == manifest['architecture']['hiddenSize']
    assert config['layer_types'] == manifest['inference']['layerPattern']['layerTypes']
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    assert config['rms_norm_eps'] == manifest['inference']['normalization']['rmsNormEps']
    reference = import_tool('independent_qwen', contract['inputs']['equations']['path'])
    weights = RetainedWeights(root, manifest, inventory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in contract['cases']:
        assert all(type(i) is int and 0 <= i < config['vocab_size'] for i in case['inputIds'])
        assert all(0 <= i < len(case['inputIds']) for i in case['positions'])
        start = time.monotonic()
        layers = []
        def progress(row):
            layers.append(row)
            print(json.dumps({'case': case['id'], **row}), flush=True)
        logits, caches = reference.evaluate(case['inputIds'], weights, config,
                                           np.array(case['positions']), progress)
        assert np.isfinite(logits).all()
        target = output / (case['id'] + '.npy')
        np.save(target, logits, allow_pickle=False)
        rows.append(dict(id=case['id'], inputIds=case['inputIds'], positions=case['positions'],
                         logits=dict(path=target.name, sha256=file_hash(target), shape=list(logits.shape)),
                         greedyTokens=np.argmax(logits, axis=1).tolist(), layers=layers, caches=caches,
                         elapsedSeconds=time.monotonic() - start))
    result = dict(schema='doppler.qwen-quantized-independent-reference/v1', qualified=False,
                  scope='Independent F64 equations with exact decoded retained weights and F16 KV storage; diagnostic reference, not GPU qualification',
                  contractSha256=file_hash(contract_path), toolSha256=file_hash(__file__),
                  manifestSha256=inventory['manifestSha256'], cases=rows, tensorReads=weights.reads,
                  maxRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  threads={key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']})
    (output / 'reference.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.contract, args.model, args.out)
