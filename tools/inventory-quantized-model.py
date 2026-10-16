#!/usr/bin/env python3
"""Freeze retained RDRR bytes and independent Q4_K block observations; no conversion."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def spans_for(entry):
    return entry.get('spans') or [dict(shardIndex=entry['shard'], offset=entry['offset'], size=entry['size'])]


def tensor_chunks(root, manifest, entry, start=0, size=None):
    remaining = entry['size'] - start if size is None else size
    assert 0 <= start <= entry['size'] and 0 <= remaining <= entry['size'] - start
    cursor = 0
    for span in spans_for(entry):
        end = cursor + span['size']
        if start < end and remaining:
            offset = max(start - cursor, 0)
            count = min(span['size'] - offset, remaining)
            shard = manifest['shards'][span['shardIndex']]
            with (root / shard['filename']).open('rb') as stream:
                stream.seek(span['offset'] + offset)
                while count:
                    block = stream.read(min(count, 2 * 1024 * 1024))
                    assert block, 'Truncated tensor span'
                    yield block
                    count -= len(block)
                    remaining -= len(block)
                    start += len(block)
        cursor = end
    assert remaining == 0, 'Incomplete tensor span mapping'


def inventory(root, source_revision):
    import numpy as np
    from gguf import GGMLQuantizationType, dequantize

    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['artifactIdentity']['sourceRevision'] == source_revision
    assert manifest['artifactIdentity']['sourceCheckpointId'] == 'Qwen/Qwen3.8-27B'
    files = []
    for shard in manifest['shards']:
        path = (root / shard['filename']).resolve()
        assert path.parent == root and path.stat().st_size == shard['size']
        files.append(dict(path=shard['filename'], bytes=shard['size'], sha256=sha_file(path),
                          declaredHash=shard['hash']))
    assets = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha_file(p))
              for p in sorted(root.iterdir()) if p.is_file() and p.suffix in ('.json', '.txt')]
    tensors, samples, totals = [], [], {}
    sample_names = {'lm_head.weight', 'model.language_model.layers.0.mlp.gate_proj.weight',
                    'model.language_model.layers.0.linear_attn.in_proj_qkv.weight'}
    for name, entry in manifest['tensors'].items():
        count = math.prod(entry['shape'])
        dtype = entry['dtype']
        assert dtype in ('F16', 'F32', 'Q4_K_M'), dtype
        expected = count * (2 if dtype == 'F16' else 4) if dtype != 'Q4_K_M' else count // 256 * 144
        assert expected == entry['size']
        if dtype == 'Q4_K_M':
            assert entry['layout'] == 'row' and entry['shape'][-1] % 256 == 0
        spans = spans_for(entry)
        assert sum(s['size'] for s in spans) == entry['size']
        for span in spans:
            assert 0 <= span['shardIndex'] < len(manifest['shards'])
            assert 0 <= span['offset'] and 0 < span['size']
            assert span['offset'] + span['size'] <= manifest['shards'][span['shardIndex']]['size']
        digest = hashlib.sha256()
        for block in tensor_chunks(root, manifest, entry):
            digest.update(block)
        tensors.append(dict(name=name, shape=entry['shape'], dtype=dtype, layout=entry.get('layout'),
                            bytes=entry['size'], sha256=digest.hexdigest(), spans=spans))
        totals[dtype] = totals.get(dtype, 0) + entry['size']
        if dtype == 'Q4_K_M' and name in sample_names:
            blocks = count // 256
            for index in sorted({0, blocks // 2, blocks - 1}):
                packed = b''.join(tensor_chunks(root, manifest, entry, index * 144, 144))
                decoded = dequantize(np.frombuffer(packed, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
                assert decoded.size == 256 and np.isfinite(decoded).all()
                samples.append(dict(tensor=name, blockIndex=index, packedHex=packed.hex(),
                                    decoded=decoded.reshape(-1).tolist()))
    return dict(schema='doppler.retained-quantized-inventory/v1', root=str(root),
                manifestSha256=sha_file(root / 'manifest.json'), source=manifest['artifactIdentity'],
                declaredShardHashAlgorithm=manifest['hashAlgorithm'], files=files, assets=assets,
                tensors=tensors, tensorBytesByDtype=totals, totalTensorBytes=sum(totals.values()),
                independentDecoder=dict(package='gguf', version=importlib.metadata.version('gguf')),
                blockSamples=samples, qualified=False,
                scope='SHA-256 inventory of unchanged retained bytes; block samples are not model qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    assert not output.exists(), 'Refusing to replace retained evidence'
    result = inventory(args.model, args.revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['manifestSha256', 'tensorBytesByDtype', 'totalTensorBytes']}))
