"""Pinned upstream QSA indexer and sparse attention; synthetic CPU fixtures, not compatibility.

The unchanged upstream definitions run at the pinned text geometry with
deterministic synthetic weights. No checkpoint weights, conversion or Doppler
execution. Run: python3 -I artifacts/qwen38-flash-next-20261013/prepare-sparse-attention-reference.py
"""
import ast
import copy
import hashlib
import json
import math
import platform
import re
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
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
cfg = SimpleNamespace(**config)
cfg._attn_implementation = 'eager'

# ---- Unchanged upstream definitions -------------------------------------------
selected = ['Qwen4ExpTextRotaryEmbedding', 'Qwen4ExpTextRMSNorm', 'rotate_half', 'apply_rotary_pos_emb',
            'Qwen4ExpTextQSAIndexer', 'repeat_kv', 'eager_attention_forward', 'Qwen4ExpTextAttention']
future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
rope_nodes = [n for n in ast.parse((ROOT / 'source/modeling_rope_utils.py').read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'dynamic_rope_update']
model_nodes = [n for n in ast.parse((ROOT / 'source/modeling_qwen4_exp.py').read_text()).body
               if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected]
assert len(rope_nodes) == 1 and sorted(n.name for n in model_nodes) == sorted(selected)
assert not any(n.decorator_list for n in rope_nodes + model_nodes)

class EagerOnlyRegistry:
    """Test stub for the attention registry: selects the extracted eager function only."""
    @staticmethod
    def get_interface(name, default):
        if name != 'eager':
            raise ValueError(f'unsupported attention implementation {name!r}')
        return default

namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'wraps': wraps, 'ALL_ATTENTION_FUNCTIONS': EagerOnlyRegistry}
for body, label in [(rope_nodes, '<pinned-rope-utils>'), (model_nodes, '<pinned-qsa-attention>')]:
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[])), label, 'exec'), namespace)
upstream_eager = namespace['eager_attention_forward']
assert torch.get_float32_matmul_precision() == 'highest'
torch.set_num_threads(2)

# ---- Deterministic dyadic synthetic values --------------------------------------
def splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

def dyadic(shape, stream, exponent):
    """Exact F32 values k * 2^(exponent-23), k uniform in [-2^23, 2^23)."""
    index = np.arange(math.prod(shape), dtype=np.uint64) + np.uint64(stream << 40)
    k = (splitmix64(index) >> np.uint64(40)).astype(np.int64) - (1 << 23)
    return torch.from_numpy((k * 2.0 ** (exponent - 23)).astype(np.float32).reshape(shape))

def digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()

LAYER = cfg.layer_types.index('full_attention')
attention = namespace['Qwen4ExpTextAttention'](cfg, LAYER).eval()
indexer = attention.indexer
exponents = {'q_proj': -5, 'k_proj': -5, 'v_proj': -5, 'o_proj': -6, 'index_qk_proj': -5}
weights = []
with torch.no_grad():
    for stream, (name, parameter) in enumerate(sorted(attention.named_parameters())):
        exponent = -3 if 'norm' in name else exponents[name.split('.')[-2]]
        parameter.copy_(dyadic(tuple(parameter.shape), stream + 1, exponent))
        weights.append({'name': name, 'shape': list(parameter.shape), 'stream': stream + 1,
                        'exponent': exponent, 'sha256': digest(parameter)})
B, T, PAD = 2, 2560, [0, 5]
hidden = dyadic((B, T, cfg.hidden_size), 101, 1)
positions = torch.arange(T).view(1, 1, T).expand(3, B, T)
rotary = namespace['Qwen4ExpTextRotaryEmbedding'](cfg)
cos, sin = rotary(hidden, positions)
assert cos.shape == (B, T, 64) and cos.dtype == torch.float32
j = torch.arange(T)
visible = torch.stack([(j[None, :] <= j[:, None]) & (j[None, :] >= p) for p in PAD]).unsqueeze(1)
MIN = torch.finfo(torch.float32).min
float_mask = torch.where(visible, 0.0, MIN)

with torch.no_grad():
    selected_bool = indexer(hidden, (cos, sin), visible, None)
    selected_float = indexer(hidden, (cos, sin), float_mask, None)
assert selected_bool.dtype == torch.bool and selected_bool.shape == visible.shape
bool_float_equal = bool(torch.equal(selected_float, torch.where(selected_bool, 0.0, MIN)))
sel = selected_bool[:, 0].numpy()
vis = visible[:, 0].numpy()

# ---- Independent float64 recomputation of the indexer -----------------------------
heads, width, ratio = cfg.indexer_n_heads, cfg.indexer_head_dim, cfg.indexer_compress_ratio
budget_blocks = cfg.indexer_budget // ratio
eps = cfg.rms_norm_eps
w_qk = indexer.index_qk_proj.weight.detach().double().numpy()
w_qn = indexer.q_layernorm.weight.detach().double().numpy()
w_kn = indexer.k_layernorm.weight.detach().double().numpy()
# Upstream angle semantics: F32 inverse frequencies and F32 position products.
rotary_dim = int(cfg.head_dim * cfg.rope_parameters['partial_rotary_factor'])
exponent = (np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)).astype(np.float32)
inv_freq = (np.float32(1) / np.power(np.float32(cfg.rope_parameters['rope_theta']), exponent)).astype(np.float32)
angle = (np.arange(T, dtype=np.float32)[:, None] * inv_freq[None, :]).astype(np.float32).astype(np.float64)
cos64 = np.concatenate([np.cos(angle)] * 2, -1)
sin64 = np.concatenate([np.sin(angle)] * 2, -1)
rotary_check = {'invFreqBitwise': bool((inv_freq == rotary.inv_freq.numpy()).all()),
                'maxCosSinDifference': float(max(np.abs(cos64 - cos[0].double().numpy()).max(),
                                                 np.abs(sin64 - sin[0].double().numpy()).max())),
                'maxF32AngleEffect': float(np.abs(np.cos(np.arange(T)[:, None] * inv_freq.astype(np.float64)) - np.cos(angle)).max())}
far = np.arange(cfg.max_position_embeddings - 4096, cfg.max_position_embeddings, dtype=np.float32)[:, None]
rotary_check['maxF32AngleEffectNearMaxPosition'] = float(np.abs(
    np.cos(far.astype(np.float64) * inv_freq.astype(np.float64)) - np.cos((far * inv_freq).astype(np.float32).astype(np.float64))).max())

def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (1 + w)

def rope(x, c, s):
    r = x[..., :rotary_dim]
    turned = np.concatenate([-r[..., rotary_dim // 2:], r[..., :rotary_dim // 2]], -1)
    return np.concatenate([r * c + turned * s, x[..., rotary_dim:]], -1)

qk = hidden.double().numpy() @ w_qk.T
q = rope(rms(qk[..., :heads * width].reshape(B, T, heads, width), w_qn), cos64[None, :, None], sin64[None, :, None])
raw = qk[..., heads * width:]

def block_start(b, variant):
    # Upstream groups consecutive visible tokens; the alternative aligns blocks to absolute slots.
    return -(-PAD[b] // ratio) * ratio if variant == 'absolute-alignment' else PAD[b]

def block_scores(b, variant=None):
    start = block_start(b, variant)
    count = (T - start) // ratio
    tokens = start + ratio * np.arange(count)[:, None] + np.arange(ratio)
    keys = raw[b, tokens]
    if variant == 'norm-then-mean':
        pooled = rms(keys, w_kn).mean(1)
    elif variant == 'no-key-norm':
        pooled = keys.mean(1)
    else:
        pooled = rms(keys.mean(1), w_kn)
    at = tokens[:, -1] if variant == 'rope-last-token' else tokens[:, 0]
    pooled = rope(pooled, cos64[at], sin64[at])
    dots = np.einsum('thd,nd->thn', q[b], pooled)
    if variant == 'no-relu':
        return tokens, dots.sum(1) / math.sqrt(width)
    if variant == 'max-heads':
        return tokens, np.maximum(dots, 0).max(1) / math.sqrt(width)
    return tokens, np.maximum(dots, 0).sum(1) / math.sqrt(width)

def expected_rows(b, variant=None):
    """Exact expected selection per query, or the tie-invariant parts when the boundary is tied."""
    tokens, scores = block_scores(b, variant)
    rows = {}
    start = block_start(b, variant)
    for t in range(T):
        lead = np.arange(PAD[b], min(start, t + 1))
        body = np.arange(start, t + 1)
        complete = len(body) // ratio
        tail = np.concatenate([lead, body[complete * ratio:]])
        if complete <= budget_blocks:
            rows[t] = {'kind': 'dense', 'blocks': np.arange(complete), 'tail': tail}
            continue
        s = scores[t, :complete]
        order = np.argsort(-s, kind='stable')
        kth, next_ = s[order[budget_blocks - 1]], s[order[budget_blocks]]
        if kth > next_:
            rows[t] = {'kind': 'sparse', 'blocks': np.sort(order[:budget_blocks]), 'tail': tail,
                       'gap': float(kth - next_), 'scale': float(s.max())}
        else:
            rows[t] = {'kind': 'tied', 'above': np.nonzero(s > kth)[0], 'tied': np.nonzero(s == kth)[0],
                       'below': np.nonzero(s < kth)[0], 'tail': tail, 'value': float(kth)}
    return tokens, rows

def mask_row(tokens, blocks, tail):
    row = np.zeros(T, dtype=bool)
    row[tokens[blocks].reshape(-1)] = True
    row[tail] = True
    return row

summary, sparse_gaps = [], []
tie_choice = {'atZero': 0, 'lowestIndices': 0, 'highestIndices': 0, 'neither': 0}
reference_rows = {}
for b in range(B):
    tokens, rows = expected_rows(b)
    counts = {'dense': 0, 'sparse': 0, 'tied': 0}
    for t, row in rows.items():
        counts[row['kind']] += 1
        actual = sel[b, t]
        assert not (actual & ~vis[b, t]).any(), 'selected a token outside the supplied mask'
        if row['kind'] != 'tied':
            assert (actual == mask_row(tokens, row['blocks'], row['tail'])).all(), (b, t, row['kind'])
            if row['kind'] == 'sparse':
                sparse_gaps.append(row['gap'] / row['scale'])
            continue
        chosen = [n for n in row['tied'] if actual[tokens[n]].all()]
        partial = [n for n in row['tied'] if actual[tokens[n]].any() and not actual[tokens[n]].all()]
        assert not partial and all(actual[tokens[n]].all() for n in row['above'])
        assert not any(actual[tokens[n]].any() for n in row['below']) and actual[row['tail']].all()
        assert len(chosen) == budget_blocks - len(row['above'])
        tie_choice['atZero'] += row['value'] == 0
        tie_choice['lowestIndices' if chosen == list(row['tied'][:len(chosen)]) else
                   'highestIndices' if chosen == list(row['tied'][len(row['tied']) - len(chosen):]) else 'neither'] += 1
    first_sparse = next(t for t in range(T) if rows[t]['kind'] != 'dense')
    self_excluded = [t for t in range(PAD[b], T) if not sel[b, t, t]]
    summary.append({'row': b, 'leftPadding': PAD[b], 'queries': counts, 'firstSparseQuery': first_sparse,
                    'visibleAtFirstSparseQuery': first_sparse - PAD[b] + 1,
                    'selectionSha256': hashlib.sha256(np.packbits(sel[b]).tobytes()).hexdigest(),
                    'denseQueriesEqualVisible': bool(all((sel[b, t] == vis[b, t]).all() for t in range(first_sparse))),
                    'selfTokenExcludedQueries': len(self_excluded), 'firstSelfTokenExcluded': self_excluded[:4],
                    'zeroScoreBlocksAtLastQuery': int((block_scores(b)[1][T - 1, :(T - PAD[b]) // ratio] == 0).sum())})
    reference_rows[b] = (tokens, rows)

# Alternative semantics: does this fixture distinguish them from the upstream contract?
alternatives = {}
for variant in ['rope-last-token', 'norm-then-mean', 'no-key-norm', 'no-relu', 'max-heads', 'absolute-alignment']:
    changed = 0
    for b in range(B):
        tokens, rows = reference_rows[b]
        alt_tokens, alt_rows = expected_rows(b, variant)
        for t, row in rows.items():
            if row['kind'] == 'tied' or alt_rows[t]['kind'] == 'tied':
                continue
            if not (mask_row(tokens, row['blocks'], row['tail']) == mask_row(alt_tokens, alt_rows[t]['blocks'], alt_rows[t]['tail'])).all():
                changed += 1
    alternatives[variant] = {'queriesChanged': changed, 'distinguished': changed > 0}

# ---- Sparse attention through an explicit cache port, chunked prefill and decode ----------
class CachePort:
    """Explicit test port: appends keys, values and raw indexer keys; not upstream DynamicCache."""
    def __init__(self, layer_idx):
        self.layer_idx, self.keys, self.values, self.raw = layer_idx, None, None, None
    def update(self, keys, values, layer_idx):
        assert layer_idx == self.layer_idx
        self.keys = keys if self.keys is None else torch.cat([self.keys, keys], -2)
        self.values = values if self.values is None else torch.cat([self.values, values], -2)
        return self.keys, self.values
    def update_indexer(self, raw_keys, layer_idx):
        assert layer_idx == self.layer_idx
        self.raw = raw_keys if self.raw is None else torch.cat([self.raw, raw_keys], 1)
        return self.raw

SAMPLES = [0, 3, 4, 2047, 2050, 2051, 2052, 2300, 2556, 2557, 2558, 2559]
CHUNKS = [1024, 1024, 509, 1, 1, 1]
captured = {'indexer': [], 'eager': {}, 'linear': {}}
window = {}

def keep(rows_tensor, start):
    return {start + i: rows_tensor[i] for i in range(rows_tensor.shape[0]) if start + i in SAMPLES}

def eager_spy(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    output, probabilities = upstream_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
    start = window['start']
    for t in range(query.shape[2]):
        if start + t in SAMPLES:
            captured['eager'][start + t] = {
                'query': query[0, :, t].clone(), 'key': key[0].clone(), 'value': value[0].clone(),
                'mask': attention_mask[0, 0, t].clone(), 'scaling': scaling,
                'output': output[0, t].clone(), 'probabilities': probabilities[0, :, t].clone()}
    return output, probabilities

namespace['eager_attention_forward'] = eager_spy
hooks = [attention.indexer.register_forward_hook(lambda m, a, out: captured['indexer'].append(out[0, 0] == 0))]
for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    def hook(module, args, out, name=name):
        captured['linear'].setdefault(name, {}).update(
            {t: (args[0][0, i].clone(), out[0, i].clone()) for i, t in enumerate(range(window['start'], window['start'] + out.shape[1])) if t in SAMPLES})
    hooks.append(getattr(attention, name).register_forward_hook(hook))
port = CachePort(LAYER)
outputs, start = [], 0
with torch.no_grad():
    for size in CHUNKS:
        end = start + size
        window['start'] = start
        out, _ = attention(hidden[:1, start:end], (cos[:1, :end], sin[:1, :end]), float_mask[:1, :, start:end, :end], port)
        outputs.append(out)
        start = end
for h in hooks:
    h.remove()
namespace['eager_attention_forward'] = upstream_eager
assert start == T and port.raw.shape == (1, T, width)
chunk_rows = [row for part in captured['indexer'] for row in part]
chunked_selection_equal = all(np.array_equal(r.numpy(), sel[0, t, :r.shape[0]]) and not sel[0, t, r.shape[0]:].any()
                              for t, r in enumerate(chunk_rows))

# ---- Derived F32 bounds for the sampled attention core and projections ------------------
U, C_EXP, UP, T64 = 2.0 ** -24, 4, 1 + 2.0 ** -40, 2.0 ** -53
gamma = lambda n: n * U / (1 - n * U)
L = lambda t: -math.log1p(-t)
ATTENTION_DERIVATION = [
    'Eager core: S = fl(q.k)*scaling with scaling = 2^-4 exact; |S~ - S| <= beta = gamma_256 * scaling * sum|q||k|.',
    'Masked entries add finfo.min (or -inf after min+min) and exponentiate to exactly 0, so they contribute nothing.',
    'Softmax (PyTorch _vec_softmax_lastdim): fl(S~ - m~), exp within C_EXP*u, Z summed over the row in any order '
    '(gamma_{n-1}), R = fl(1/Z), p = fl(E*R). Without renormalization R and Z are not cancelled: '
    '|log(p/pi)| <= N_j + D with N_j = beta_j + u(|S_j-m| + beta_j + beta_max) + L(C_EXP u) + 2L(u), '
    'D = max_k(beta_k + u(|S_k-m| + beta_k + beta_max)) + L(C_EXP u) + L(gamma_{n-1}).',
    'Output: |o~ - sum pi v| <= gamma_n sum p~|v| + sum pi expm1(N_j+D)|v|; n is the key length including masked zeros.',
    'Projections: |y~ - xW^T| <= gamma_fanIn |x||W|^T. All bounds add float64 truth allowances and (1 + 2^-40) outward rounding.',
]

def check_linear(name, fan_in):
    weight = getattr(attention, name).weight.detach().double().numpy()
    worst = 0.0
    for t, (x, y) in captured['linear'][name].items():
        x, y = x.double().numpy(), y.double().numpy()
        magnitude = np.abs(weight) @ np.abs(x)
        bound = (gamma(fan_in) + fan_in * T64) * magnitude * UP
        worst = max(worst, float((np.abs(y - weight @ x) / bound).max()))
    return {'samples': len(captured['linear'][name]), 'fanIn': fan_in, 'maxRatio': worst, 'passed': worst <= 1}

group = cfg.num_attention_heads // cfg.num_key_value_heads
core = {'samples': [], 'maxProbabilityRatio': 0.0, 'maxOutputRatio': 0.0, 'maskComposition': True, 'maskedProbabilitiesZero': True}
for t in SAMPLES:
    c = captured['eager'][t]
    n = c['key'].shape[1]
    attended = (c['mask'] == 0).numpy()
    expected_attended = sel[0, t, :n] & vis[0, t, :n]
    core['maskComposition'] &= bool((attended == expected_attended).all() and (c['mask'][~torch.from_numpy(attended)] <= MIN).all())
    qh = c['query'].double().numpy()
    kh = np.repeat(c['key'].double().numpy(), group, 0)[:, attended]
    vh = np.repeat(c['value'].double().numpy(), group, 0)[:, attended]
    p = c['probabilities'].double().numpy()
    core['maskedProbabilitiesZero'] &= bool((p[:, ~attended] == 0).all())
    p = p[:, attended]
    S = np.einsum('hd,hnd->hn', qh, kh) * c['scaling']
    beta = (gamma(cfg.head_dim) + 2 * cfg.head_dim * T64) * c['scaling'] * np.einsum('hd,hnd->hn', np.abs(qh), np.abs(kh)) * UP
    m = S.max(1, keepdims=True)
    shift = U * (np.abs(S - m) + beta + beta.max(1, keepdims=True))
    pi = np.exp(S - m)
    pi /= pi.sum(1, keepdims=True)
    N = beta + shift + L(C_EXP * U) + 2 * L(U)
    D = (beta + shift).max(1, keepdims=True) + L(C_EXP * U) + L(gamma(n - 1))
    dp = pi * np.expm1(N + D) * UP + 16 * T64 * pi
    prob_ratio = float((np.abs(p - pi) / dp).max())
    exact_out = np.einsum('hn,hnd->hd', pi, vh)
    out_bound = (gamma(n) * np.einsum('hn,hnd->hd', p, np.abs(vh)) + np.einsum('hn,hnd->hd', dp, np.abs(vh))
                 + 2 * n * T64 * np.einsum('hn,hnd->hd', pi, np.abs(vh))) * UP
    out_ratio = float((np.abs(c['output'].double().numpy() - exact_out) / out_bound).max())
    core['maxProbabilityRatio'] = max(core['maxProbabilityRatio'], prob_ratio)
    core['maxOutputRatio'] = max(core['maxOutputRatio'], out_ratio)
    core['samples'].append({'query': t, 'keyLength': n, 'attended': int(attended.sum()),
                            'probabilityRatio': prob_ratio, 'outputRatio': out_ratio,
                            'outputSha256': digest(c['output'])})
core['passed'] = core['maskComposition'] and core['maskedProbabilitiesZero'] and core['maxProbabilityRatio'] <= 1 and core['maxOutputRatio'] <= 1
projections = {'q_proj': check_linear('q_proj', cfg.hidden_size), 'k_proj': check_linear('k_proj', cfg.hidden_size),
               'v_proj': check_linear('v_proj', cfg.hidden_size),
               'o_proj': check_linear('o_proj', cfg.num_attention_heads * cfg.head_dim)}

# ---- Malformed and silently accepted inputs (single decode query through the port) -------
def decode_query(position_cos, position_sin, mask, hidden_row=None):
    probe = CachePort(LAYER)
    probe.update_indexer(port.raw[:, :T - 1], LAYER)
    row = hidden[:1, T - 1:] if hidden_row is None else hidden_row
    with torch.no_grad():
        return indexer(row, (position_cos, position_sin), mask, probe)

reference_decode = decode_query(cos[:1], sin[:1], visible[:1, :, T - 1:])
assert torch.equal(reference_decode[0, 0, 0], selected_bool[0, 0, T - 1])
rejections = {}
for label, call in [
        ('hiddenWidth2559', lambda: decode_query(cos[:1], sin[:1], visible[:1, :, T - 1:], hidden[:1, T - 1:, :-1])),
        ('maskLongerThanKeys', lambda: decode_query(cos[:1], sin[:1], torch.ones(1, 1, 1, T + 4, dtype=torch.bool))),
        ('positionsShorterThanKeys', lambda: decode_query(cos[:1, :T - 8], sin[:1, :T - 8], visible[:1, :, T - 1:]))]:
    try:
        call()
        rejections[label] = {'rejected': False}
    except (RuntimeError, IndexError, ValueError) as error:
        rejections[label] = {'rejected': True, 'error': type(error).__name__}
extra_positions = torch.arange(T + 8).view(1, 1, -1).expand(3, 1, -1)
long_cos, long_sin = rotary(hidden[:1], extra_positions)
accepted = {
    'positionsLongerThanKeys': decode_query(long_cos, long_sin, visible[:1, :, T - 1:]),
    'nonCausalMaskSelectsFutureTokens': None,
    'maskShorterThanKeys': decode_query(cos[:1], sin[:1], visible[:1, :, T - 1:, :T - 8]),
}
with torch.no_grad():
    noncausal = indexer(hidden[:1, :8], (cos[:1, :8], sin[:1, :8]), torch.ones(1, 1, 8, 8, dtype=torch.bool), None)
silent = {
    'positionsLongerThanKeys': {'accepted': True, 'selectionChanged': not torch.equal(accepted['positionsLongerThanKeys'], reference_decode),
                                'note': 'query RoPE uses the last positions while block keys index positions from slot 0'},
    'nonCausalMask': {'accepted': True, 'futureTokensSelected': bool(noncausal[0, 0, 0, 1:].any()),
                      'note': 'the indexer inherits causality entirely from the supplied mask'},
    'maskShorterThanKeys': {'accepted': True, 'outputColumns': int(accepted['maskShorterThanKeys'].shape[-1]),
                            'note': 'keys beyond the mask width are ignored without error'},
}

# ---- Observation: checkpoint dtype changes selection --------------------------------
bf16 = copy.deepcopy(indexer).to(torch.bfloat16)
hidden_bf16 = hidden[:1].to(torch.bfloat16)
cos_bf16, sin_bf16 = rotary(hidden_bf16, positions[:, :1])
with torch.no_grad():
    sel_bf16 = bf16(hidden_bf16, (cos_bf16, sin_bf16), visible[:1], None)[0, 0].numpy()
sparse_queries = [t for t, row in reference_rows[0][1].items() if row['kind'] != 'dense']
bf16_changed = sum(not np.array_equal(sel_bf16[t], sel[0, t]) for t in sparse_queries)

SEMANTICS = {
    'establishes': [
        'Selection draws only from the supplied mask; a query with at most 2051 visible tokens selects exactly its visible set (dense).',
        'Blocks are consecutive groups of 4 visible tokens starting at the first visible token; with left padding they are not aligned to absolute slots.',
        'The trailing (visible mod 4) most recent tokens are always selected; complete blocks compete for 512 slots, at most 2051 tokens per query.',
        'Block key = k_layernorm(mean of 4 raw pre-norm, pre-RoPE indexer keys), RoPE at the block first-token slot; query = q_layernorm then RoPE at its own slot.',
        'Block score = sum over the 4 indexer heads of relu(q.k), divided by sqrt(128); RoPE covers the first 64 of 128 indexer dims.',
        'The query own token is not guaranteed: when visible mod 4 == 0 its block can lose top-k (see selfTokenExcludedQueries).',
        'Bool and 0/finfo.min float masks give the same selection; eager attention receives causal + selection and masked probabilities are exactly 0.',
        'Attention uses uncompressed per-token keys and values of the selected tokens; pooling exists only in the indexer score.',
        'Prefill chunks and single-token decode through an appending raw-key cache give the same selection as one full-sequence call.',
        'Sampled eager attention probabilities, outputs and q/k/v/o projections are within derived F32 bounds of float64 truth.',
    ],
    'doesNotEstablish': [
        'Tie order: when the 512th and 513th block scores are equal (relu zeros; see tieBoundaryChoice.atZero) torch.topk order is '
        'unspecified; only the tie-invariant parts are checked.',
        'Near-tie decidability: untied boundaries with tiny relative gaps matched here, but another correct F32 order could flip them; '
        'an implementation-independent acceptance rule for near ties is open.',
        'Checkpoint dtype: the config dtype is bfloat16; selection under bfloat16 differs on many sparse queries (observations).',
        'create_causal_mask construction, sdpa/flex dispatch, DynamicCache.update_indexer internals and generate-style padded position ids.',
        'Multimodal MRoPE positions (the three text position streams are equal here) and positions beyond 2560, where F32 angle rounding grows.',
        'Numerics of q/k norms, RoPE and the sigmoid output gate as separate stages; learned weights; linear-attention, MTP and PLE layers.',
    ],
}
checks = {
    'indexerBoolFloatMaskEquivalent': bool_float_equal,
    'indexerMatchesIndependentFloat64': True,  # assertions in the comparison loop raise on any mismatch
    'chunkedPrefillDecodeMatchesFullSequence': chunked_selection_equal,
    'attentionCoreWithinDerivedBound': core['passed'],
    'projectionsWithinDerivedBound': all(p['passed'] for p in projections.values()),
    'malformedInputsRejected': all(r['rejected'] for r in rejections.values()),
    'rotaryAnglesBitwiseReplicated': rotary_check['invFreqBitwise'],
}
report = {
    'schema': 'doppler.qwen38-sparse-attention-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU',
    'environment': {'machine': platform.machine(), 'blas': (re.search(r'BLAS_INFO=([^,\s]+)', torch.__config__.show()) or [None, None])[1],
                    'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'threads': torch.get_num_threads()},
    'scope': 'Unchanged upstream QSA indexer, rotary embedding, RMSNorm and eager sparse attention at the pinned text geometry; '
             'synthetic F32 weights; no checkpoint weights, conversion or Doppler execution',
    'sourceNodes': {'modeling_qwen4_exp.py': selected, 'modeling_rope_utils.py': ['dynamic_rope_update']},
    'testPorts': ['EagerOnlyRegistry selects the extracted eager_attention_forward', 'CachePort appends keys, values and raw indexer keys',
                  'eager_attention_forward spy records sampled rows and returns the upstream result unchanged'],
    'geometry': {'layer': LAYER, 'hiddenSize': cfg.hidden_size, 'heads': cfg.num_attention_heads, 'kvHeads': cfg.num_key_value_heads,
                 'headDim': cfg.head_dim, 'rotaryDim': rotary_dim, 'indexerHeads': heads, 'indexerKvHeads': cfg.indexer_kv_heads,
                 'indexerHeadDim': width, 'indexerBudget': cfg.indexer_budget, 'compressRatio': ratio, 'budgetBlocks': budget_blocks,
                 'maxSelectedTokens': cfg.indexer_budget + ratio - 1, 'sequence': T, 'batch': B},
    'recipe': {'values': 'splitmix64(index + stream<<40) >> 40, minus 2^23, times 2^(exponent-23); exact F32',
               'weights': weights, 'hidden': {'stream': 101, 'exponent': 1, 'sha256': digest(hidden)},
               'positions': 'arange(sequence) for all three MRoPE streams and both rows', 'leftPadding': PAD,
               'masks': 'causal & key >= padding; bool for the sdpa-style path, 0/finfo.min float for eager'},
    'semantics': SEMANTICS,
    'rotary': rotary_check,
    'indexer': {'rows': summary, 'tieBoundaryChoice': tie_choice,
                'minRelativeBoundaryGap': float(min(sparse_gaps)),
                'untiedSparseQueriesByRelativeGap': {f'below{limit:g}': int(sum(g < limit for g in sparse_gaps))
                                                     for limit in [1e-5, 1e-4, 1e-3, 1e-2]},
                'sampleExcludedBlocks': {str(t): sorted(set(range(t // ratio)) - set(reference_rows[0][1][t]['blocks'].tolist()))
                                         for t in SAMPLES if reference_rows[0][1][t]['kind'] == 'sparse'}},
    'alternativeSemantics': alternatives,
    'sparseAttention': {'chunks': CHUNKS, 'derivation': ATTENTION_DERIVATION, 'core': core, 'projections': projections,
                        'outputSha256': digest(torch.cat(outputs, 1))},
    'malformedRejected': rejections, 'acceptedWithoutValidation': silent,
    'observations': {'bfloat16SparseQueriesWithDifferentSelection': bf16_changed, 'sparseQueries': len(sparse_queries),
                     'note': 'upstream indexer run in bfloat16 with the same synthetic values; dtype sensitivity only, not a reference'},
    'checks': checks,
}
report['passed'] = all(checks.values())
report['modelQualified'] = False
(ROOT / 'sparse-attention-reference.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks, 'rows': summary, 'tieChoice': tie_choice,
                  'alternatives': alternatives, 'core': {k: core[k] for k in ['maxProbabilityRatio', 'maxOutputRatio']},
                  'bf16Changed': bf16_changed, 'modelQualified': False}, default=str))
if not report['passed']:
    raise SystemExit(1)
