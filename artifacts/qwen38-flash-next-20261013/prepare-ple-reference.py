"""Pinned upstream PLE layer (Qwen4ExpTextPLELayer) on synthetic weights; not compatibility.

Executes the unchanged upstream PLE layer, n-gram embedding and RMSNorm at the pinned
geometry (4 x 2560 streams, 16 heads x 160 rows, kernel 4 dilation 3) with deterministic
dyadic weights. The 95 GiB table is replaced by a row spy: row(address)[d] is the BF16-exact value
((splitmix64(address * 160 + d + (207 << 40)) >> 56) - 128) * 2^(exponent - 7), so a consumer regenerates
any row exactly and the rows carry the checkpoint's stored precision.
Token history and short-conv state go through an explicit cache port that follows the pinned
LinearAttentionLayer.update_conv_state semantics (record_past disabled). Runs the full
sequence and a chunked schedule; captures the internal sigmoid gate input and conv output.
Run: python3 -I artifacts/qwen38-flash-next-20261013/prepare-ple-reference.py
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
for entry in binding['files'] + binding['referenceImplementation']['files']:
    assert hashlib.sha256((ROOT / 'source' / entry['path']).read_bytes()).hexdigest() == entry['sha256']
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
config_class = next(n for n in ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
config.setdefault('seed', next(ast.literal_eval(n.value) for n in config_class.body
                               if isinstance(n, ast.AnnAssign) and n.target.id == 'seed'))
cfg = SimpleNamespace(**config)
cfg.ple_embed_dim = cfg.ple_embed_dim or cfg.hidden_size

selected = {'Qwen4ExpTextRMSNorm', 'apply_mask_to_padding_states', '_splitmix64', '_build_layer_multipliers',
            '_is_prime', '_find_nth_prime_after', 'Qwen4ExpTextNGramEmbedding', 'Qwen4ExpTextPLELayer'}
constants = {'_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2', '_PRIME_1'}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
for node in ast.parse((ROOT / 'source/modeling_qwen4_exp.py').read_text()).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected:
        assert not node.decorator_list
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
        nodes.append(node)

def splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

def dyadic_at(index, stream, exponent):
    k = (splitmix64(index.astype(np.uint64) + np.uint64(stream << 40)) >> np.uint64(40)).astype(np.int64) - (1 << 23)
    return (k * 2.0 ** (exponent - 23)).astype(np.float32)

def bf16_rows(index, stream, exponent):
    k = (splitmix64(index.astype(np.uint64) + np.uint64(stream << 40)) >> np.uint64(56)).astype(np.int64) - 128
    return (k * 2.0 ** (exponent - 7)).astype(np.float32)

def dyadic(shape, stream, exponent):
    return torch.from_numpy(dyadic_at(np.arange(math.prod(shape), dtype=np.uint64), stream, exponent).reshape(shape))

ROW_STREAM, ROW_EXPONENT = 207, 0
row_width = cfg.ple_embed_dim // ((cfg.ngram_size - 1) * cfg.heads_per_ngram)
lookups = []

class RowSpy(nn.Module):
    """Stands in for the 51B-parameter embedding: deterministic rows by address."""
    def __init__(self, count, width):
        super().__init__()
        assert width == row_width
        self.count = count
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    def forward(self, indices):
        assert int(indices.max()) < self.count and int(indices.min()) >= 0
        lookups.append(indices.clone())
        flat = indices.reshape(-1).numpy().astype(np.uint64)
        grid = flat[:, None] * np.uint64(row_width) + np.arange(row_width, dtype=np.uint64)[None, :]
        return torch.from_numpy(bf16_rows(grid, ROW_STREAM, ROW_EXPONENT)).reshape(*indices.shape, row_width)

class NNProxy:
    def __getattr__(self, name):
        return RowSpy if name == 'Embedding' else getattr(nn, name)

captured = {'gate': [], 'conv': []}
class TorchProxy:
    def __getattr__(self, name):
        if name == 'sigmoid':
            def sigmoid(x):
                captured['gate'].append(x.detach().clone())
                return torch.sigmoid(x)
            return sigmoid
        return getattr(torch, name)

class FProxy:
    def __getattr__(self, name):
        if name == 'silu':
            def silu(x):
                captured['conv'].append(x.detach().clone())
                return F.silu(x)
            return silu
        return getattr(F, name)

namespace = {'torch': TorchProxy(), 'nn': NNProxy(), 'F': FProxy(), 'math': math, 'Cache': object}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-ple>', 'exec'), namespace)
torch.set_num_threads(2)

class CachePort:
    """Explicit test port for conv_states[1] (short conv) and [2] (token history), following the
    pinned LinearAttentionLayer.update_conv_state with record_past disabled. Not DynamicCache."""
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.layers = {layer_idx: SimpleNamespace(conv_states={1: None, 2: None})}
        self.has = {1: False, 2: False}
        self.size = {}
    def has_previous_state(self, layer_idx, state_idx):
        assert layer_idx == self.layer_idx
        return self.has[state_idx]
    def update_conv_state(self, states, layer_idx, state_idx, conv_kernel_size):
        assert layer_idx == self.layer_idx
        layer = self.layers[layer_idx]
        if layer.conv_states[state_idx] is None:
            self.size[state_idx] = conv_kernel_size
            layer.conv_states[state_idx] = torch.zeros((*states.shape[:-1], conv_kernel_size), dtype=states.dtype)
        if not self.has[state_idx]:
            full = states
            self.has[state_idx] = True
            if full.shape[-1] < self.size[state_idx]:
                full = F.pad(full, (self.size[state_idx] - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([layer.conv_states[state_idx], states], dim=-1)
        layer.conv_states[state_idx].copy_(full[..., -self.size[state_idx]:])
        return full

LAYER = cfg.ple_layer_ids[0] - 1
ple = namespace['Qwen4ExpTextPLELayer'](cfg, LAYER, 0).eval()
streams = {'key_proj.weight': (201, -6), 'value_proj.weight': (202, -6), 'norm_key.weight': (203, -3),
           'norm_query.weight': (204, -3), 'norm_conv.weight': (205, -3), 'conv1d.weight': (206, -2)}
weights = []
with torch.no_grad():
    for name, parameter in ple.named_parameters():
        if name.startswith('ple_embedding.'):
            continue
        stream, exponent = streams[name]
        parameter.copy_(dyadic(tuple(parameter.shape), stream, exponent))
        weights.append({'name': name, 'shape': list(parameter.shape), 'stream': stream, 'exponent': exponent,
                        'sha256': hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()})
EOS = cfg.eos_token_id
tokens = [17, 42, EOS, 101, 102, 103, EOS, 248319, 5, 6, 7, EOS, EOS, 9, 10, 11, 12, 13, EOS, 14, 15, 16, 17, 18]
T, HC = len(tokens), cfg.hc_count * cfg.hidden_size
HIDDEN_STREAM, HIDDEN_EXPONENT = 208, 1
hidden = dyadic((1, T, HC), HIDDEN_STREAM, HIDDEN_EXPONENT)
ids = torch.tensor([tokens])

def run(chunks):
    port = CachePort(LAYER)
    outputs, start = [], 0
    lookups.clear()
    captured['gate'].clear()
    captured['conv'].clear()
    with torch.no_grad():
        for size in chunks:
            outputs.append(ple(hidden[:, start:start + size], ids[:, start:start + size], port))
            start += size
    assert start == T
    return (torch.cat(outputs, 1), torch.cat(lookups, 1), torch.cat(captured['gate'], 1),
            torch.cat([c.transpose(1, 2) for c in captured['conv']], 1), port)

full, addresses, gate, conv, port = run([T])
CHUNKS = [5, 1, 1, 17]
chunked, chunk_addresses, chunk_gate, chunk_conv, chunk_port = run(CHUNKS)
assert torch.equal(addresses, chunk_addresses), 'addresses differ between full and chunked runs'
STRIDE = 16
channels = list(range(0, HC, STRIDE))
out = {
    'schema': 'doppler.qwen38-ple-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU',
    'scope': 'Unchanged upstream Qwen4ExpTextPLELayer on dyadic synthetic weights; the n-gram table is a row spy; '
             'cache state through an explicit port; no learned weights, Doppler execution or padding masks.',
    'geometry': {'hcCount': cfg.hc_count, 'hiddenSize': cfg.hidden_size, 'embedDim': cfg.ple_embed_dim,
                 'heads': (cfg.ngram_size - 1) * cfg.heads_per_ngram, 'rowWidth': row_width,
                 'convKernel': cfg.ple_conv_kernel_size, 'convDilation': cfg.ngram_size,
                 'convState': (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size, 'eps': cfg.rms_norm_eps,
                 'decoderLayerIndex': LAYER},
    'recipe': {'values': 'splitmix64(index + stream<<40) >> 40, minus 2^23, times 2^(exponent-23); exact F32',
               'rows': {'stream': ROW_STREAM, 'exponent': ROW_EXPONENT, 'index': 'address * rowWidth + d',
                        'values': '((splitmix64(index + stream<<40) >> 56) - 128) * 2^(exponent-7); exact BF16'},
               'hidden': {'stream': HIDDEN_STREAM, 'exponent': HIDDEN_EXPONENT, 'shape': [T, HC]},
               'weights': weights},
    'tokens': tokens,
    'chunks': CHUNKS,
    'addresses': addresses[0].tolist(),
    'gateInput': gate[0, :, :, 0].tolist(),
    'outputChannels': {'stride': STRIDE, 'count': len(channels)},
    'output': full[0][:, channels].tolist(),
    'chunkedOutputMaxAbsDifference': float((chunked - full).abs().max()),
    'convOutputChunkedMaxAbsDifference': float((chunk_conv - conv).abs().max()),
    'finalConvState': port.layers[LAYER].conv_states[1][0][channels].tolist(),
    'finalTokenHistory': port.layers[LAYER].conv_states[2][0].tolist(),
    'chunkedTokenHistoryEqual': bool(torch.equal(port.layers[LAYER].conv_states[2], chunk_port.layers[LAYER].conv_states[2])),
    'chunkedConvStateMaxAbsDifference': float((port.layers[LAYER].conv_states[1] - chunk_port.layers[LAYER].conv_states[1]).abs().max()),
}
target = ROOT / 'ple-reference.json'
target.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(json.dumps({k: out[k] for k in ['chunkedOutputMaxAbsDifference', 'convOutputChunkedMaxAbsDifference', 'chunkedTokenHistoryEqual', 'chunkedConvStateMaxAbsDifference']}
                 | {'bytes': target.stat().st_size, 'gate0': out['gateInput'][0]}))
