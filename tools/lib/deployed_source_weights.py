"""Verified RDRR weights for offline source-equation controls, never Run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode_tensor(data, entry):
    assert len(data) == entry['size']
    dtype = entry['dtype']
    if dtype == 'Q4_K_M':
        assert entry['layout'] == 'row'
        array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
    elif dtype == 'BF16':
        array = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    else:
        assert dtype in ('F16', 'F32'), dtype
        array = np.frombuffer(data, dtype='<f2' if dtype == 'F16' else '<f4').astype(np.float32)
    assert array.size == np.prod(entry['shape']) and np.isfinite(array).all()
    return torch.from_numpy(array.copy().reshape(entry['shape']))


def load_deployed_weights(model, root, expected, source_receipt):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    assert sha256(manifest_path) == expected['manifestSha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['modelId'] == expected['modelId']
    identity = manifest['artifactIdentity']
    assert identity['sourceRepo'] == source_receipt['source']
    assert identity['sourceRevision'] == source_receipt['revision']
    assert manifest['hashAlgorithm'] == 'sha256'
    assert manifest['inference']['session']['kvcache']['kvDtype'] == 'f16'
    wanted = model.state_dict()
    assert set(wanted) == set(manifest['tensors']), 'Deployed tensors must exactly cover source parameters.'
    shards, identities = [], []
    for index, shard in enumerate(manifest['shards']):
        assert shard['index'] == index and Path(shard['filename']).name == shard['filename']
        path = root / shard['filename']
        assert path.stat().st_size == shard['size'] and sha256(path) == shard['hash']
        shards.append(path)
        identities.append({'filename': shard['filename'], 'bytes': shard['size'], 'sha256': shard['hash']})
    weights = {}
    for name, entry in manifest['tensors'].items():
        assert list(wanted[name].shape) == entry['shape'], name
        spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
        parts = []
        for span in spans:
            assert 0 <= span['shardIndex'] < len(shards)
            assert span['offset'] >= 0 and span['size'] > 0
            path = shards[span['shardIndex']]
            assert span['offset'] + span['size'] <= path.stat().st_size
            with path.open('rb') as stream:
                stream.seek(span['offset'])
                parts.append(stream.read(span['size']))
        weights[name] = decode_tensor(b''.join(parts), entry)
    model.load_state_dict(weights, strict=True, assign=True)
    # A tied source checkpoint may be materialized as distinct deployed precisions.
    # Strictly load both deployed tensors; never overwrite the head by retying it.
    assert model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
    model.eval()
    return {'manifestSha256': expected['manifestSha256'], 'modelId': manifest['modelId'],
            'weights': identities, 'tensorCount': len(weights), 'finiteDecodedTensors': True,
            'independentDecoder': 'gguf.dequantize Q4_K; NumPy IEEE F16/F32/BF16',
            'headRetiedToEmbedding': False}
