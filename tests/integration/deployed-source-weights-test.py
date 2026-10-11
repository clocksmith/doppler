"""Offline loader controls; synthetic tensors are not model qualification."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/lib'))
from deployed_source_weights import decode_tensor, load_deployed_weights


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(1, 256)
        self.lm_head = torch.nn.Linear(256, 1, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens


digest = lambda data: hashlib.sha256(data).hexdigest()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    embedding = np.ones(256, dtype='<f2').tobytes()
    head = bytes(144)  # One independently decoded zero Q4_K block.
    shard = embedding + head
    (root / 'weights.bin').write_bytes(shard)
    source = {'source': 'fixture', 'revision': 'revision'}
    manifest = {
        'modelId': 'fixture-deployed', 'hashAlgorithm': 'sha256',
        'artifactIdentity': {'sourceRepo': 'fixture', 'sourceRevision': 'revision'},
        'inference': {'session': {'kvcache': {'kvDtype': 'f16'}}},
        'shards': [{'index': 0, 'filename': 'weights.bin', 'size': len(shard), 'hash': digest(shard)}],
        'tensors': {
            'embed_tokens.weight': {'shard': 0, 'offset': 0, 'size': len(embedding), 'shape': [1, 256], 'dtype': 'F16'},
            'lm_head.weight': {'spans': [{'shardIndex': 0, 'offset': len(embedding), 'size': len(head)}],
                               'size': len(head), 'shape': [1, 256], 'dtype': 'Q4_K_M', 'layout': 'row'},
        },
    }

    def load(value):
        data = json.dumps(value).encode()
        (root / 'manifest.json').write_bytes(data)
        model = Model()
        result = load_deployed_weights(model, root,
                                       {'modelId': 'fixture-deployed', 'manifestSha256': digest(data)}, source)
        return model, result

    model, result = load(manifest)
    assert result['finiteDecodedTensors'] and not result['headRetiedToEmbedding']
    assert torch.equal(model.embed_tokens.weight, torch.ones(1, 256))
    assert torch.equal(model.lm_head.weight, torch.zeros(1, 256))
    for kind in ['shape', 'layout', 'span', 'hash', 'source', 'missing']:
        bad = copy.deepcopy(manifest)
        if kind == 'shape': bad['tensors']['lm_head.weight']['shape'] = [2, 128]
        if kind == 'layout': bad['tensors']['lm_head.weight']['layout'] = 'column'
        if kind == 'span': bad['tensors']['lm_head.weight']['spans'][0]['size'] += 1
        if kind == 'hash': bad['shards'][0]['hash'] = '0' * 64
        if kind == 'source': bad['artifactIdentity']['sourceRevision'] = 'other'
        if kind == 'missing': del bad['tensors']['lm_head.weight']
        try:
            load(bad)
            raise RuntimeError(f'{kind} drift was not rejected')
        except AssertionError:
            pass
    for dtype, data in [('F16', np.array([float('inf')], dtype='<f2').tobytes()),
                        ('F32', np.array([float('nan')], dtype='<f4').tobytes())]:
        try:
            decode_tensor(data, {'size': len(data), 'dtype': dtype, 'shape': [1]})
            raise RuntimeError('Nonfinite weight was not rejected')
        except AssertionError:
            pass
print(json.dumps({'passed': True, 'distinctQuantizedHeadPreserved': True,
                  'invalidArtifactControls': 8, 'modelQualification': False}))
