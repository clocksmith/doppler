"""Pinned upstream decoder-layer composition (hyper-connections, PLE, injection, root mixer); synthetic.

Runs the unchanged upstream Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, PLE layer and
n-gram embedding for decoder layers 0-3 (linear, linear + PLE, linear, full attention) followed by
the root hyper-connection mixer, at the pinned 4 x 2560 stream geometry. The attention, linear
attention and MoE branch modules are replaced by deterministic low-rank linear spies
(y = (x A) B, A [2560, 16], B [16, 2560], dyadic) because their arithmetic has separate pinned
references; this fixture establishes composition order only. Entry replicates the embeddings
over the streams exactly as Qwen4ExpTextModel.forward does (hidden_states.repeat(1, 1, hc_count)).
The n-gram table is the BF16-exact row spy of prepare-ple-reference.py. Full and chunked
(5, 1, 1, 17) schedules run through explicit cache ports.
Run: python3 -I artifacts/qwen38-flash-next-20261013/prepare-decoder-composition-reference.py
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
model_source = (ROOT / 'source/modeling_qwen4_exp.py').read_text()
assert 'hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)' in model_source
assert 'hidden_states = self.hyper_connection_mixer(hidden_states)' in model_source

selected = {'Qwen4ExpTextRMSNorm', 'apply_mask_to_padding_states', '_splitmix64', '_build_layer_multipliers', '_is_prime',
            '_find_nth_prime_after', 'Qwen4ExpTextNGramEmbedding', 'Qwen4ExpTextPLELayer', 'Qwen4ExpTextGatedResidual',
            'Qwen4ExpTextDecoderLayer'}
constants = {'_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2', '_PRIME_1'}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
for node in ast.parse(model_source).body:
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

def dyadic(shape, stream, exponent):
    index = np.arange(math.prod(shape), dtype=np.uint64) + np.uint64(stream << 40)
    k = (splitmix64(index) >> np.uint64(40)).astype(np.int64) - (1 << 23)
    return torch.from_numpy((k * 2.0 ** (exponent - 23)).astype(np.float32).reshape(shape))

def bf16_rows(index, stream, exponent):
    k = (splitmix64(index.astype(np.uint64) + np.uint64(stream << 40)) >> np.uint64(56)).astype(np.int64) - 128
    return (k * 2.0 ** (exponent - 7)).astype(np.float32)

ROW_STREAM, ROW_EXPONENT = 207, 0
row_width = cfg.ple_embed_dim // ((cfg.ngram_size - 1) * cfg.heads_per_ngram)
RANK = 16

class RowSpy(nn.Module):
    def __init__(self, count, width):
        super().__init__()
        self.count = count
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    def forward(self, indices):
        flat = indices.reshape(-1).numpy().astype(np.uint64)
        grid = flat[:, None] * np.uint64(row_width) + np.arange(row_width, dtype=np.uint64)[None, :]
        return torch.from_numpy(bf16_rows(grid, ROW_STREAM, ROW_EXPONENT)).reshape(*indices.shape, row_width)

class BranchSpy(nn.Module):
    """Deterministic low-rank linear branch: y = (x A) B. Not an upstream module."""
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.A = nn.Parameter(torch.empty(config.hidden_size, RANK), requires_grad=False)
        self.B = nn.Parameter(torch.empty(RANK, config.hidden_size), requires_grad=False)
        self.calls = 0
    def branch(self, x):
        self.calls += 1
        return (x @ self.A) @ self.B

class LinearSpy(BranchSpy):
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return self.branch(hidden_states)

class AttentionSpy(BranchSpy):
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        return self.branch(hidden_states), None

class MoESpy(BranchSpy):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, hidden_states):
        return self.branch(hidden_states)

class NNProxy:
    def __getattr__(self, name):
        return RowSpy if name == 'Embedding' else getattr(nn, name)

namespace = {'torch': torch, 'nn': NNProxy(), 'F': F, 'math': math, 'Cache': object, 'TransformersKwargs': dict,
             'Unpack': type('Unpack', (), {'__class_getitem__': classmethod(lambda cls, item: item)}),
             'GradientCheckpointingLayer': nn.Module, 'Qwen4ExpTextGatedDeltaNet': LinearSpy,
             'Qwen4ExpTextAttention': AttentionSpy, 'Qwen4ExpTextSparseMoeBlock': MoESpy}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-decoder>', 'exec'), namespace)
torch.set_num_threads(2)

class CachePort:
    """Explicit port for PLE conv_states[1] (short conv) and [2] (token history); pinned update_conv_state semantics."""
    def __init__(self):
        self.layers, self.has, self.size = {}, {}, {}
    def has_previous_state(self, layer_idx, state_idx):
        return self.has.get((layer_idx, state_idx), False)
    def update_conv_state(self, states, layer_idx, state_idx, conv_kernel_size):
        layer = self.layers.setdefault(layer_idx, SimpleNamespace(conv_states={}))
        key = (layer_idx, state_idx)
        if state_idx not in layer.conv_states:
            self.size[key] = conv_kernel_size
            layer.conv_states[state_idx] = torch.zeros((*states.shape[:-1], conv_kernel_size), dtype=states.dtype)
        if not self.has.get(key, False):
            full = states
            self.has[key] = True
            if full.shape[-1] < self.size[key]:
                full = F.pad(full, (self.size[key] - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([layer.conv_states[state_idx], states], dim=-1)
        layer.conv_states[state_idx].copy_(full[..., -self.size[key]:])
        return full

LAYERS = [0, 1, 2, 3]
layers = [namespace['Qwen4ExpTextDecoderLayer'](cfg, i).eval() for i in LAYERS]
mixer = namespace['Qwen4ExpTextGatedResidual'](cfg, use_combine=False).eval()
assert layers[1].ple is not None and all(layers[i].ple is None for i in (0, 2, 3))
assert [layer.layer_type for layer in layers] == ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
records, stream_id = [], 300
with torch.no_grad():
    for prefix, module in [*[(f'layers.{i}.', layer) for i, layer in zip(LAYERS, layers)], ('mixer.', mixer)]:
        for name, parameter in module.named_parameters():
            if 'ngram_embedding' in name:
                continue
            exponent = -3 if ('norm' in name) else -2 if 'conv1d' in name else -6
            stream_id += 1
            parameter.copy_(dyadic(tuple(parameter.shape), stream_id, exponent))
            records.append({'name': prefix + name, 'shape': list(parameter.shape), 'stream': stream_id, 'exponent': exponent,
                            'sha256': hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()})
EOS = cfg.eos_token_id
tokens = [17, 42, EOS, 101, 102, 103, EOS, 248319, 5, 6, 7, EOS, EOS, 9, 10, 11, 12, 13, EOS, 14, 15, 16, 17, 18]
T = len(tokens)
EMBED_STREAM, EMBED_EXPONENT = 299, 0
embeds = dyadic((1, T, cfg.hidden_size), EMBED_STREAM, EMBED_EXPONENT)
ids = torch.tensor([tokens])

def run(chunks):
    port, outputs, layer_streams, start = CachePort(), [], {i: [] for i in LAYERS}, 0
    with torch.no_grad():
        for size in chunks:
            hidden = embeds[:, start:start + size].repeat(1, 1, cfg.hc_count)
            for i, layer in zip(LAYERS, layers):
                hidden = layer(hidden, position_embeddings=None, attention_mask=None, conv_mask=None,
                               past_key_values=port, ple_input_ids=ids[:, start:start + size])
                layer_streams[i].append(hidden)
            outputs.append(mixer(hidden))
            start += size
    return torch.cat(outputs, 1), {i: torch.cat(v, 1) for i, v in layer_streams.items()}

full, full_streams = run([T])
CHUNKS = [5, 1, 1, 17]
chunked, chunk_streams = run(CHUNKS)
STRIDE = 16
HC = cfg.hc_count * cfg.hidden_size
out = {
    'schema': 'doppler.qwen38-decoder-composition-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU',
    'scope': 'Unchanged upstream decoder-layer composition, gated residuals, PLE and root mixer for layers 0-3 on dyadic '
             'synthetic weights; attention, linear attention and MoE are low-rank linear spies (composition order only); '
             'no learned weights, masks or Doppler execution.',
    'layers': LAYERS, 'branchRank': RANK,
    'recipe': {'values': 'splitmix64(index + stream<<40) >> 40, minus 2^23, times 2^(exponent-23); exact F32',
               'rows': {'stream': ROW_STREAM, 'exponent': ROW_EXPONENT,
                        'values': '((splitmix64(address * rowWidth + d + stream<<40) >> 56) - 128) * 2^(exponent-7)'},
               'embeds': {'stream': EMBED_STREAM, 'exponent': EMBED_EXPONENT, 'shape': [T, cfg.hidden_size]},
               'parameters': records},
    'tokens': tokens, 'chunks': CHUNKS,
    'outputChannels': {'stride': STRIDE},
    'mixedOutput': full[0][:, ::STRIDE].tolist(),
    'layerStreams': {str(i): full_streams[i][0][:, ::STRIDE * 4].tolist() for i in LAYERS},
    'chunkedMaxAbsDifference': float((chunked - full).abs().max()),
    'chunkedLayerMaxAbsDifference': {str(i): float((chunk_streams[i] - full_streams[i]).abs().max()) for i in LAYERS},
}
target = ROOT / 'decoder-composition-reference.json'
target.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(json.dumps({'bytes': target.stat().st_size, 'chunked': out['chunkedMaxAbsDifference'],
                  'layers': out['chunkedLayerMaxAbsDifference'], 'mixed0': out['mixedOutput'][0][:3]}))
