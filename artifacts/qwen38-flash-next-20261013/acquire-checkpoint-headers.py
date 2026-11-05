"""Bounded SafeTensors header and PLE buffer acquisition at the pinned revision.

Reads only header bytes of all 131 checkpoint files and the three small PLE
buffers (layer_multipliers, ngram_heads_offsets, ngram_heads_vocab_sizes). No
weight row is downloaded. The n-gram table row mapping is derived from the
pinned loader: `Concatenate(dim=0)` over `ngram_embedding.shard_*.weight`, with
keys ordered by the pinned `dot_natural_key` (executed here, not restated).
Run from the Doppler root: python3 -I artifacts/qwen38-flash-next-20261013/acquire-checkpoint-headers.py
"""
import ast
import hashlib
import json
import math
import re
import struct
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
for entry in binding['files'] + binding['referenceImplementation']['files']:
    assert hashlib.sha256((ROOT / 'source' / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
loader_entry = next(e for e in binding['referenceImplementation']['files'] if e['path'] == 'core_model_loading.py')
mapping_entry = next(e for e in binding['referenceImplementation']['files'] if e['path'] == 'conversion_mapping.py')
REVISION = binding['revision']
BASE = f"https://huggingface.co/{binding['model']}/resolve/{REVISION}/"
MAX_HEADER = 4 << 20

# Upstream definitions, executed unchanged.
modeling = ast.parse((ROOT / 'source/modeling_qwen4_exp.py').read_text())
selected = {'_splitmix64', '_build_layer_multipliers', '_is_prime', '_find_nth_prime_after'}
constants = {'_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2', '_PRIME_1'}
nodes = []
for node in modeling.body:
    if isinstance(node, ast.FunctionDef) and node.name in selected:
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
        nodes.append(node)
loading = ast.parse((ROOT / 'source/core_model_loading.py').read_text())
nodes += [n for n in loading.body if isinstance(n, ast.FunctionDef) and n.name == 'dot_natural_key']
mapping_source = (ROOT / 'source/conversion_mapping.py').read_text()
assert 'source_patterns="ngram_embedding.shard_*.weight"' in mapping_source
assert 'operations=[Concatenate(dim=0, num_shards_attribute="split_ngram_parts")]' in mapping_source


class LongTensorStandIn(list):
    """Python integers in place of torch.long; upstream only constructs the tensor."""


namespace = {'math': math, 're': re, 'torch': SimpleNamespace(tensor=lambda values, dtype=None: LongTensorStandIn(values), long='long')}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned>', 'exec'), namespace)

config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
config_class = next(n for n in ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
defaults = {n.target.id: ast.literal_eval(n.value) for n in config_class.body
            if isinstance(n, ast.AnnAssign) and n.value is not None and isinstance(n.target, ast.Name)
            and n.target.id in {'seed', 'norm_topk_prob'}}
assert 'seed' not in config and 'norm_topk_prob' not in config
seed = defaults['seed']
ngram_heads = (config['ngram_size'] - 1) * config['heads_per_ngram']
ple_layer_index = 0
sizes, offsets, total = [], [], 0
for head in range(ngram_heads):
    size = namespace['_find_nth_prime_after'](config['ngram_vocab_size_base'] - 1, ple_layer_index * ngram_heads + head + 1)
    sizes.append(size)
    offsets.append(total)
    total += size
padded = math.ceil(total / config['make_ngram_vocab_size_divisible_by']) * config['make_ngram_vocab_size_divisible_by']
multipliers = list(namespace['_build_layer_multipliers'](config['vocab_size'], config['ngram_size'], ple_layer_index, seed))


def fetch(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'doppler-source-preparation'})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 206, (url, response.status)
        body = response.read(end - start + 2)
    assert len(body) == end - start + 1, (url, len(body))
    return body


index_bytes = (ROOT / 'source/model.safetensors.index.json').read_bytes()
weight_map = json.loads(index_bytes)['weight_map']
files = sorted(set(weight_map.values()))
headers = {}
for name in files:
    url = BASE + name
    length = struct.unpack('<Q', fetch(url, 0, 7))[0]
    assert 0 < length <= MAX_HEADER, (name, length)
    raw = fetch(url, 8, 8 + length - 1)
    parsed = json.loads(raw)
    tensors = {k: v for k, v in parsed.items() if k != '__metadata__'}
    expected = sorted(k for k, f in weight_map.items() if f == name)
    assert sorted(tensors) == expected, f'{name}: header tensors differ from the index'
    headers[name] = {
        'headerBytes': length,
        'headerSha256': hashlib.sha256(raw).hexdigest(),
        'dataStart': 8 + length,
        'metadata': parsed.get('__metadata__'),
        'tensors': {k: {'dtype': v['dtype'], 'shape': v['shape'], 'dataOffsets': v['data_offsets']}
                    for k, v in sorted(tensors.items())},
    }

prefix = 'model.language_model.layers.1.ple.ple_embedding.'
shard_keys = [k for k in weight_map if k.startswith(prefix + 'ngram_embedding.shard_')]
assert len(shard_keys) == config['split_ngram_parts']
ordered = sorted(shard_keys, key=namespace['dot_natural_key'])
ELEMENT_BYTES = {'BF16': 2}
shards, row = [], 0
for position, key in enumerate(ordered):
    assert key == f'{prefix}ngram_embedding.shard_{position}.weight'
    entry = headers[weight_map[key]]['tensors'][key]
    rows, width = entry['shape']
    assert entry['dtype'] in ELEMENT_BYTES
    begin, end = entry['dataOffsets']
    assert end - begin == rows * width * ELEMENT_BYTES[entry['dtype']]
    shards.append({
        'shard': position, 'tensor': key, 'file': weight_map[key], 'dtype': entry['dtype'], 'shape': entry['shape'],
        'rowBegin': row, 'rowEnd': row + rows,
        'fileByteBegin': headers[weight_map[key]]['dataStart'] + begin,
        'fileByteEnd': headers[weight_map[key]]['dataStart'] + end,
    })
    row += rows
assert row == padded, (row, padded)
assert {s['shape'][1] for s in shards} == {config['ple_embed_dim'] // ngram_heads}
assert {s['dtype'] for s in shards} == {'BF16'}

buffers = {}
for field, expected_values in (('layer_multipliers', multipliers), ('ngram_heads_offsets', offsets),
                               ('ngram_heads_vocab_sizes', sizes)):
    key = prefix + field
    name = weight_map[key]
    entry = headers[name]['tensors'][key]
    assert entry['dtype'] == 'I64', (key, entry['dtype'])
    begin, end = entry['dataOffsets']
    raw = fetch(BASE + name, headers[name]['dataStart'] + begin, headers[name]['dataStart'] + end - 1)
    values = list(struct.unpack(f'<{len(raw) // 8}q', raw))
    assert entry['shape'] == [len(values)]
    assert values == expected_values, f'{field}: checkpoint buffer differs from the pinned derivation'
    buffers[field] = {'tensor': key, 'file': name, 'dtype': 'I64', 'shape': entry['shape'],
                      'sha256': hashlib.sha256(raw).hexdigest(), 'values': [str(v) for v in values]}

out = {
    'schema': 'doppler.qwen4-exp-checkpoint-headers/v1',
    'model': binding['model'],
    'revision': REVISION,
    'indexSha256': hashlib.sha256(index_bytes).hexdigest(),
    'loader': {'concatenate': f"{mapping_entry['url']} (Concatenate(dim=0) over ngram_embedding.shard_*.weight)",
               'ordering': f"{loader_entry['url']} dot_natural_key"},
    'resolvedSourceDefaults': {'seed': seed, 'norm_topk_prob': defaults['norm_topk_prob']},
    'ngramTable': {
        'decoderLayerIndex': 1, 'sourceLayerId': config['ple_layer_ids'][0], 'pleLayerIndex': ple_layer_index,
        'rows': padded, 'unpaddedRows': total, 'rowWidth': config['ple_embed_dim'] // ngram_heads, 'dtype': 'BF16',
        'shardRows': sorted({s['rowEnd'] - s['rowBegin'] for s in shards}), 'shards': shards,
    },
    'buffers': buffers,
    'files': headers,
}
target = ROOT / 'checkpoint-headers.json'
target.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({'files': len(headers), 'shards': len(shards), 'rows': padded,
                  'shardRows': out['ngramTable']['shardRows'], 'buffers': {k: v['values'] for k, v in buffers.items()}}))
