"""Stage-level QSA indexer reference for Doppler operator tests; synthetic weights, not compatibility.

Reuses the sparse-attention fixture recipe (verified by its retained SHA256s) and the
unchanged upstream Qwen4ExpTextQSAIndexer, run through an explicit appending cache port
in chunks (1024, 1024, 509, 1, 1, 1) on row 0 (no padding). Emits, for sampled queries,
the upstream selected tokens; float64 block scores; and first-order F32 error bounds for
any summation order, split as A + B * eps where eps bounds the absolute error of the
executing implementation's cos/sin tables (measured by the consumer). Also emits float64
query vectors and block keys with bounds of the same form.
Run: python3 -I artifacts/qwen38-flash-next-20261013/prepare-qsa-stage-reference.py
"""
import ast
import hashlib
import json
import math
import platform
from functools import wraps
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
prior = json.loads((ROOT / 'sparse-attention-reference.json').read_text())
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
cfg = SimpleNamespace(**config)
cfg._attn_implementation = 'eager'

selected = ['Qwen4ExpTextRotaryEmbedding', 'Qwen4ExpTextRMSNorm', 'rotate_half', 'apply_rotary_pos_emb',
            'Qwen4ExpTextQSAIndexer', 'repeat_kv', 'eager_attention_forward', 'Qwen4ExpTextAttention']
future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
rope_nodes = [n for n in ast.parse((ROOT / 'source/modeling_rope_utils.py').read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'dynamic_rope_update']
model_nodes = [n for n in ast.parse((ROOT / 'source/modeling_qwen4_exp.py').read_text()).body
               if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected]
assert len(rope_nodes) == 1 and sorted(n.name for n in model_nodes) == sorted(selected)
namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'wraps': wraps, 'ALL_ATTENTION_FUNCTIONS': None}
for body, label in [(rope_nodes, '<pinned-rope-utils>'), (model_nodes, '<pinned-qsa-attention>')]:
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[])), label, 'exec'), namespace)
torch.set_num_threads(2)

def splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

def dyadic(shape, stream, exponent):
    index = np.arange(math.prod(shape), dtype=np.uint64) + np.uint64(stream << 40)
    k = (splitmix64(index) >> np.uint64(40)).astype(np.int64) - (1 << 23)
    return torch.from_numpy((k * 2.0 ** (exponent - 23)).astype(np.float32).reshape(shape))

def digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()

LAYER = cfg.layer_types.index('full_attention')
attention = namespace['Qwen4ExpTextAttention'](cfg, LAYER).eval()
indexer = attention.indexer
recipe = {w['name']: w for w in prior['recipe']['weights']}
with torch.no_grad():
    for name, parameter in sorted(attention.named_parameters()):
        spec = recipe[name]
        parameter.copy_(dyadic(tuple(parameter.shape), spec['stream'], spec['exponent']))
        assert digest(parameter) == spec['sha256'], name
T = prior['geometry']['sequence']
hidden = dyadic((1, T, cfg.hidden_size), prior['recipe']['hidden']['stream'], prior['recipe']['hidden']['exponent'])
positions = torch.arange(T).view(1, 1, T).expand(3, 1, T)
rotary = namespace['Qwen4ExpTextRotaryEmbedding'](cfg)
cos, sin = rotary(hidden, positions)
MIN = torch.finfo(torch.float32).min

class CachePort:
    """Explicit test port: appends raw indexer keys; not upstream DynamicCache."""
    def __init__(self):
        self.raw = None
    def update_indexer(self, raw_keys, layer_idx):
        assert layer_idx == LAYER
        self.raw = raw_keys if self.raw is None else torch.cat([self.raw, raw_keys], 1)
        return self.raw

CHUNKS = [1024, 1024, 509, 1, 1, 1]
SAMPLES = [0, 3, 4, 2047, 2050, 2051, 2052, 2053, 2300, 2556, 2557, 2558, 2559]
port, start, upstream_rows = CachePort(), 0, {}
j = torch.arange(T)
with torch.no_grad():
    for size in CHUNKS:
        end = start + size
        mask = torch.where(j[None, :end] <= j[start:end, None], 0.0, MIN)[None, None]
        rows = indexer(hidden[:, start:end], (cos[:, :end], sin[:, :end]), mask, port)
        for i in range(size):
            if start + i in SAMPLES:
                upstream_rows[start + i] = torch.nonzero(rows[0, 0, i] == 0).flatten().tolist()
        start = end
assert start == T and port.raw.shape == (1, T, cfg.indexer_head_dim)

# ---- float64 truth and first-order F32 bounds (any summation order) ----------------
U = 2.0 ** -24
gamma = lambda n: n * U / (1 - n * U)
heads, width, ratio = cfg.indexer_n_heads, cfg.indexer_head_dim, cfg.indexer_compress_ratio
budget_blocks, eps = cfg.indexer_budget // ratio, cfg.rms_norm_eps
rotary_dim = int(cfg.head_dim * cfg.rope_parameters['partial_rotary_factor'])
half = rotary_dim // 2
h64 = hidden[0].double().numpy()
w_qk = indexer.index_qk_proj.weight.detach().double().numpy()
w_qn = indexer.q_layernorm.weight.detach().double().numpy()
w_kn = indexer.k_layernorm.weight.detach().double().numpy()
exponent = (np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)).astype(np.float32)
inv_freq = (np.float32(1) / np.power(np.float32(cfg.rope_parameters['rope_theta']), exponent)).astype(np.float32)
assert (inv_freq == rotary.inv_freq.numpy()).all()
angle = (np.arange(T, dtype=np.float32)[:, None] * inv_freq[None, :]).astype(np.float32).astype(np.float64)
c64, s64 = np.cos(angle), np.sin(angle)  # [T, half]

proj = h64 @ w_qk.T                                  # [T, (heads + 1) * width]
proj_err = gamma(cfg.hidden_size) * (np.abs(h64) @ np.abs(w_qk).T)

def rms_with_bound(x, err, w):
    """RMSNorm (1 + w) over the last axis with propagated and rounding error."""
    sq = (x * x).sum(-1, keepdims=True)
    r = 1.0 / np.sqrt(sq / x.shape[-1] + eps)
    y = x * r * (1 + w)
    relative = (np.abs(x) * err).sum(-1, keepdims=True) / sq      # |dr / r| to first order
    rounding = 0.5 * (gamma(x.shape[-1] + 1) + 2 * U) + 2 * U + 3 * U
    bound = np.abs(1 + w) * r * (err + np.abs(x) * relative) + np.abs(y) * rounding
    return y, bound

def rope_with_bound(y, err, c, s):
    """Rotate-half on the first rotary_dim dims: returns value, A (eps-free) and B (eps coefficient)."""
    out, a, b = y.copy(), err.copy(), np.zeros_like(err)
    x1, x2 = y[..., :half], y[..., half:rotary_dim]
    e1, e2 = err[..., :half], err[..., half:rotary_dim]
    out[..., :half] = x1 * c - x2 * s
    out[..., half:rotary_dim] = x2 * c + x1 * s
    for lo, hi, p, q, ep, eq in [(0, half, x1, x2, e1, e2), (half, rotary_dim, x2, x1, e2, e1)]:
        a[..., lo:hi] = np.abs(c) * ep + np.abs(s) * eq + 2 * U * (np.abs(p * c) + np.abs(q * s)) + U * np.abs(out[..., lo:hi])
        b[..., lo:hi] = np.abs(p) + np.abs(q)
    return out, a, b

qh = proj[:, :heads * width].reshape(T, heads, width)
qh_err = proj_err[:, :heads * width].reshape(T, heads, width)
qn, qn_err = rms_with_bound(qh, qh_err, w_qn)
q, q_a, q_b = rope_with_bound(qn, qn_err, c64[:, None, :], s64[:, None, :])
raw, raw_err = proj[:, heads * width:], proj_err[:, heads * width:]
blocks = T // ratio
grouped, grouped_err = raw[:blocks * ratio].reshape(blocks, ratio, width), raw_err[:blocks * ratio].reshape(blocks, ratio, width)
pooled = grouped.mean(1)
pooled_err = grouped_err.sum(1) / ratio + (gamma(ratio + 1)) * np.abs(grouped).sum(1) / ratio
kn, kn_err = rms_with_bound(pooled, pooled_err, w_kn)
starts = np.arange(blocks) * ratio
k, k_a, k_b = rope_with_bound(kn, kn_err, c64[starts], s64[starts])
divisor = np.float64(np.float32(math.sqrt(width)))

def scores_with_bound(t):
    complete = (t + 1) // ratio
    dots = np.einsum('hd,nd->hn', q[t], k[:complete])
    mag = np.einsum('hd,nd->hn', np.abs(q[t]), np.abs(k[:complete]))
    da = gamma(width) * mag + np.einsum('hd,nd->hn', q_a[t], np.abs(k[:complete])) + np.einsum('hd,nd->hn', np.abs(q[t]), k_a[:complete])
    db = np.einsum('hd,nd->hn', q_b[t], np.abs(k[:complete])) + np.einsum('hd,nd->hn', np.abs(q[t]), k_b[:complete])
    relu = np.maximum(dots, 0)
    score = relu.sum(0) / divisor
    a = (da.sum(0) + gamma(heads + 1) * relu.sum(0)) / divisor + U * score
    b = db.sum(0) / divisor
    return score, a, b

samples = []
for t in SAMPLES:
    visible = t + 1
    complete = visible // ratio
    entry = {'query': t, 'visible': visible, 'completeBlocks': complete, 'upstreamSelected': upstream_rows[t]}
    if complete <= budget_blocks:
        entry['kind'] = 'dense'
        assert upstream_rows[t] == list(range(visible))
    else:
        score, a, b = scores_with_bound(t)
        order = np.argsort(-score, kind='stable')
        entry.update({'kind': 'sparse', 'scores': score.tolist(), 'boundA': a.tolist(), 'boundB': b.tolist(),
                      'kthScore': float(score[order[budget_blocks - 1]]), 'nextScore': float(score[order[budget_blocks]])})
    samples.append(entry)

query_samples = [{'token': t, 'head': h, 'values': q[t, h].tolist(), 'boundA': q_a[t, h].tolist(), 'boundB': q_b[t, h].tolist()}
                 for t in [0, 2052, 2559] for h in range(heads)]
key_samples = [{'block': n, 'values': k[n].tolist(), 'boundA': k_a[n].tolist(), 'boundB': k_b[n].tolist()}
               for n in [0, 1, 255, 511, 512, 639]]
out = {
    'schema': 'doppler.qwen38-qsa-stage-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU', 'machine': platform.machine(),
    'scope': 'Synthetic dyadic weights from sparse-attention-reference.json recipe, row 0 (no padding); '
             'upstream selections from the unchanged indexer via an appending cache port; float64 truth and '
             'first-order F32 bounds (A + B * eps, eps = cos/sin table absolute error). No learned weights.',
    'recipe': 'sparse-attention-reference.json#recipe (weights verified by SHA256)',
    'chunks': CHUNKS,
    'invFreqF32': [float(v) for v in inv_freq],
    'geometry': {'sequence': T, 'hiddenSize': cfg.hidden_size, 'queryHeads': heads, 'headDim': width,
                 'rotaryDim': rotary_dim, 'compressRatio': ratio, 'blockTopK': budget_blocks,
                 'denseVisibleLimit': cfg.indexer_budget + ratio - 1, 'eps': eps, 'scoreDivisorF32': float(divisor)},
    'boundDerivation': [
        'Projection: gamma_2560 |h||W|^T (inputs exact F32).',
        'RMSNorm (1 + w): propagated |1+w| r (e_i + |x_i| sum|x||e| / sum x^2) plus 0.5(gamma_{n+1} + 2u) + 5u relative rounding.',
        'Block pooling: sum of raw-key errors / ratio plus gamma_{ratio+1} sum|x| / ratio.',
        'RoPE: |c| e_p + |s| e_q + 2u(|p c| + |q s|) + u|out| + eps(|p| + |q|) for table error eps.',
        'Scores: gamma_128 |q||k| + |e_q||k| + |q||e_k| per head; relu Lipschitz 1; head sum gamma_{heads+1}; division u|s|.',
    ],
    'samples': samples,
    'querySamples': query_samples,
    'blockKeySamples': key_samples,
}
target = ROOT / 'qsa-stage-reference.json'
target.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(json.dumps({'samples': [(s['query'], s['kind'], len(s['upstreamSelected'])) for s in samples],
                  'bytes': target.stat().st_size}))
