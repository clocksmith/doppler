"""Pinned upstream gated-delta linear attention at Flash-Next geometry; synthetic CPU fixtures, not compatibility.

Runs the unchanged Qwen4ExpTextGatedDeltaNet (36 of 48 layers) with the pinned
LinearAttentionLayer cache over full-sequence, chunked-prefill and single-token
paths. Every stage is checked against float64 truth with a derived F32 bound.
Run: python3 -I artifacts/qwen38-flash-next-20261013/prepare-linear-attention-reference.py
"""
import ast
import difflib
import hashlib
import json
import math
import platform
import re
from abc import ABC, abstractmethod
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
assert torch.get_float32_matmul_precision() == 'highest'
torch.set_num_threads(2)
torch.manual_seed(0)

def source_nodes(path, names, decorators, method_decorators=None):
    """Unchanged definitions; the listed kernel/accelerate decorators are asserted and removed.

    No fla, causal_conv1d or kernels package is installed, so the pinned decorators resolve
    to these same torch bodies; accelerate hooks only act when a module has an _hf_hook."""
    found = {}
    for node in ast.parse((ROOT / 'source' / path).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            assert [ast.unparse(d) for d in node.decorator_list] == decorators.get(node.name, []), node.name
            node.decorator_list = []
            for method, expected in (method_decorators or {}).get(node.name, {}).items():
                member = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == method)
                assert [ast.unparse(d) for d in member.decorator_list] == expected, method
                member.decorator_list = []
            found[node.name] = node
    assert set(found) == set(names), set(names) - set(found)
    return [found[name] for name in names]

def run(nodes, namespace, label):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), label, 'exec'), namespace)

activations = ast.parse((ROOT / 'source/activations.py').read_text())
act2cls = next(n.value for n in activations.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ACT2CLS')
mapping = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(act2cls.keys, act2cls.values)}
assert mapping['silu'] == 'SiLUActivation' and mapping['sigmoid'] == 'nn.Sigmoid'
namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'Tensor': torch.Tensor,
             'is_torchdynamo_exporting': lambda: False, 'is_torchdynamo_compiling': lambda: False,
             'ABC': ABC, 'abstractmethod': abstractmethod}
run(source_nodes('activations.py', ['SiLUActivation'], {'SiLUActivation': ["use_kernel_forward_from_hub('SiLU')"]}), namespace, '<pinned-activations>')
namespace['ACT2FN'] = {'silu': namespace['SiLUActivation'](), 'sigmoid': nn.Sigmoid()}
run(source_nodes('cache_utils.py', ['LinearAttentionCacheLayerMixin', 'LinearAttentionLayer'], {}), namespace, '<pinned-cache-layers>')
cache_class = next(n for n in ast.parse((ROOT / 'source/cache_utils.py').read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Cache')
cache_methods = [m for m in cache_class.body if isinstance(m, ast.FunctionDef)
                 and m.name in ('update_conv_state', 'update_recurrent_state', 'has_previous_state')]
assert len(cache_methods) == 3 and not any(m.decorator_list for m in cache_methods)
run([ast.ClassDef(name='CacheMethods', bases=[], keywords=[], body=cache_methods, decorator_list=[])], namespace, '<pinned-cache-methods>')

class CachePort(namespace['CacheMethods']):
    """Test port: the three unchanged Cache methods over a list of pinned LinearAttentionLayer objects."""
    def __init__(self, layers):
        self.layers = layers
    def __len__(self):
        return len(self.layers)

GDN_NAMES = ['Qwen4ExpTextRMSNormGated', 'apply_mask_to_padding_states', 'causal_conv1d_update', 'causal_conv1d_fn', 'l2norm',
             'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule', 'Qwen4ExpTextGatedDeltaNet']
DECORATORS = {
    'Qwen4ExpTextRMSNormGated': ["use_kernel_forward_from_hub('RMSNormGated')"],
    'causal_conv1d_update': ["use_kernel_func_from_hub_with_fallback('causal_conv1d_update', 'causal_conv1d')"],
    'causal_conv1d_fn': ["use_kernel_func_from_hub_with_fallback('causal_conv1d_fn', 'causal_conv1d')"],
    'torch_chunk_gated_delta_rule': ["use_kernel_func_from_hub_with_fallback('chunk_gated_delta_rule', 'fla')"],
    'torch_recurrent_gated_delta_rule': ["use_kernel_func_from_hub_with_fallback('fused_recurrent_gated_delta_rule', 'fla')"],
    'Qwen4ExpTextGatedDeltaNet': ['use_kernelized_func([torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update])'],
}
run(source_nodes('modeling_qwen4_exp.py', GDN_NAMES, DECORATORS, {'Qwen4ExpTextGatedDeltaNet': {'forward': ["force_accelerate_hooks('conv1d')"]}}),
    namespace, '<pinned-gated-delta>')

# Qwen3.5 comparison from pinned source alone (same Transformers revision).
def segments(path, prefix):
    text = (ROOT / 'source' / path).read_text()
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            key = node.name.replace(prefix, '')
            if key in {n.replace('Qwen4ExpText', '') for n in GDN_NAMES}:
                body = '\n'.join([f'@{ast.unparse(d)}' for d in node.decorator_list] + [ast.get_source_segment(text, node)])
                out[key] = body.replace(node.name, key).replace(prefix, '')
    return out
ours, qwen35 = segments('modeling_qwen4_exp.py', 'Qwen4ExpText'), segments('modeling_qwen3_5.py', 'Qwen3_5')
qwen35_comparison = {}
for key in sorted(ours):
    diff = [line for line in difflib.unified_diff(qwen35.get(key, '').splitlines(), ours[key].splitlines(), lineterm='', n=0)
            if line[:1] in '+-' and not line.startswith(('+++', '---'))]
    qwen35_comparison[key] = {'identical': not diff, 'changedLines': diff}

# ---- Deterministic synthetic values --------------------------------------------------
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

LAYER = cfg.layer_types.index('linear_attention')
gdn = namespace['Qwen4ExpTextGatedDeltaNet'](cfg, LAYER).eval()
HK, HV, DK, DV = gdn.num_k_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
HID, KS = cfg.hidden_size, gdn.conv_kernel_size
RECIPES = {'in_proj_qkv.weight': -5, 'in_proj_z.weight': -5, 'in_proj_a.weight': -5, 'in_proj_b.weight': -5,
           'out_proj.weight': -6, 'conv1d.weight': -1}
weights = []
with torch.no_grad():
    for stream, (name, parameter) in enumerate(sorted(gdn.named_parameters())):
        if name == 'A_log':
            # A spans the upstream init range [0.01, 16] geometrically: per-token decay from 0.987 to about 1e-9.
            parameter.copy_(torch.log(0.01 * 1600.0 ** (torch.arange(HV, dtype=torch.float64) / (HV - 1))).float())
            recipe = 'log(0.01 * 1600^(h/47))'
        elif name in ('dt_bias', 'norm.weight'):
            parameter.copy_(1 + dyadic(tuple(parameter.shape), stream + 1, -3))
            recipe = f'1 + dyadic(stream {stream + 1}, exponent -3)'
        else:
            parameter.copy_(dyadic(tuple(parameter.shape), stream + 1, RECIPES[name]))
            recipe = f'dyadic(stream {stream + 1}, exponent {RECIPES[name]})'
        weights.append({'name': name, 'shape': list(parameter.shape), 'recipe': recipe, 'sha256': digest(parameter)})
T = 150
hidden = dyadic((1, T, HID), 4096, 1)

# ---- Spies: record stage inputs and outputs, return upstream results unchanged ---------
calls, current = [], {}
originals = {name: namespace[name] for name in ['causal_conv1d_fn', 'causal_conv1d_update', 'torch_chunk_gated_delta_rule',
                                                'torch_recurrent_gated_delta_rule']}
def conv_fn_spy(hidden_states, weight, bias=None, activation=None, **kwargs):
    out = originals['causal_conv1d_fn'](hidden_states, weight, bias, activation=activation, **kwargs)
    current['conv'] = ('fn', hidden_states.detach().clone(), None, out.detach().clone())
    return out
def conv_update_spy(hidden_states, conv_state, weight, bias=None, activation=None):
    before = conv_state.detach().clone()
    out = originals['causal_conv1d_update'](hidden_states, conv_state, weight, bias, activation)
    current['conv'] = ('update', hidden_states.detach().clone(), before, out.detach().clone())
    return out
def delta_spy(kind):
    def spy(query, key, value, g, beta, initial_state=None, **kwargs):
        out, final = originals[kind](query, key, value, g=g, beta=beta, initial_state=initial_state, **kwargs)
        current['core'] = {'kind': kind, 'inputs': [x.detach().clone() for x in (query, key, value, g, beta)],
                           'initial': None if initial_state is None else initial_state.detach().clone(),
                           'output': out.detach().clone(), 'final': None if final is None else final.detach().clone()}
        return out, final
    return spy
namespace.update(causal_conv1d_fn=conv_fn_spy, causal_conv1d_update=conv_update_spy,
                 torch_chunk_gated_delta_rule=delta_spy('torch_chunk_gated_delta_rule'),
                 torch_recurrent_gated_delta_rule=delta_spy('torch_recurrent_gated_delta_rule'))
for name in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']:
    getattr(gdn, name).register_forward_hook(lambda m, a, o, name=name: current.__setitem__(name, (a[0][0].detach().clone(), o[0].detach().clone())))
gdn.norm.register_forward_hook(lambda m, a, o: current.__setitem__('norm', (a[0].detach().clone(), a[1].detach().clone(), o.detach().clone())))

def forward(path, cache, start, end):
    current.clear()
    with torch.no_grad():
        out = gdn(hidden[:, start:end], cache_params=cache)
    record = dict(current, path=path, start=start, end=end, output=out[0].clone())
    if cache is not None:
        layer = cache.layers[LAYER]
        record['convState'] = layer.conv_states[0].detach().clone()
        record['recurrentState'] = layer.recurrent_states[0].detach().clone()
    calls.append(record)
    return out[0]

PATHS = {'fullSequence': [(0, T)], 'chunkedPrefillThenDecode': [(0, 70), (70, 120)] + [(t, t + 1) for t in range(120, T)],
         'shortPrefillThenDecode': [(0, 3)] + [(t, t + 1) for t in range(3, T)]}
outputs = {}
for path, spans in PATHS.items():
    cache = None if path == 'fullSequence' else CachePort([namespace['LinearAttentionLayer']()])
    outputs[path] = torch.cat([forward(path, cache, a, b) for a, b in spans])

# ---- Derived F32 bounds ---------------------------------------------------------------------
U, C_EXP, C_LOG, TINY = 2.0 ** -24, 4, 4, 2.0 ** -126
UP = 1 + 2.0 ** -28            # covers float64 truth (2^-29 of each F32 term) and bound evaluation
gamma = lambda n: n * U / (1 - n * U)
L = lambda t: -math.log1p(-t)
DENOM = (1 + C_EXP * U) * (1 + U) - 1
EPS_SIGMOID = (1 + U) / (1 - DENOM) - 1
EPS_SILU = EPS_SIGMOID
L_SILU = 1.0999
EPS_CONST = abs(float(np.float32(1e-6)) / 1e-6 - 1)
def eps_rsqrt(n):
    """Relative error of rsqrt(sum of n squares (or their mean) + eps): sum, eps add, sqrt, reciprocal."""
    theta = max(gamma(n), EPS_CONST)
    return (1 + U) / (math.sqrt(1 - theta) * math.sqrt(1 - U) * (1 - U)) - 1
EPS_L2 = (1 + eps_rsqrt(DK)) * (1 + U) - 1
SCALE_REL = abs(float(np.float32(DK ** -0.5)) / DK ** -0.5 - 1)
DIV_REL = abs(float(np.float32(DK ** 0.5)) / DK ** 0.5 - 1)
EPS_Q_CHUNK = (1 + EPS_L2) * (1 + U) * (1 + SCALE_REL) - 1
EPS_Q_RECURRENT = (1 + EPS_L2) * (1 + U) / (1 - DIV_REL) - 1
EPS_NORM = (1 + eps_rsqrt(DV)) * (1 + U) ** 3 * (1 + EPS_SIGMOID) - 1
DERIVATION = [
    'Projections: gamma_n |W||x| (n = 2560 in, 6144 out). Depthwise conv: gamma_4 sum|w||x| over the 4 taps, then silu '
    '(x / (1 + exp(-x)), Lipschitz 1.0999, relative rounding). PyTorch CPU kernels: exp within 4u, log1p within 4u, '
    'sqrt correctly rounded, rsqrt = reciprocal(sqrt).',
    'Gates: beta = 1/(1 + exp(-b)); g = -exp(A_log) * softplus(fl(a + dt_bias)), softplus = log1p(exp(x)) for x <= 20 '
    'and x above 20, Lipschitz 1.',
    'l2norm: x * rsqrt(sum x^2 + 1e-6) within (1+eps_rsqrt)(1+u); queries also scale by fl(128^-0.5) (chunk) or divide by '
    'fl(sqrt 128) (recurrent).',
    'Chunk path, per head and 64-token chunk: every operation carries |da||b| + |a||db| + |da||db| + u|computed|, '
    'matmuls gamma_n, exp e^x(e^dx(1+4u) - 1), cumsum gamma_t; the unit lower solve uses (T + F)x = b, '
    '|F| <= gamma_64|T|, giving |dx| <= (I - G)^-1 |T^-1|(|db| + (|dT| + gamma(|T|+|dT|))|x|).',
    'State error E is tracked normwise: E_next = M E + local with the exact transition M = D I - K\'^T W '
    '(chunk) or d(I - beta k k^T) (token); ||M||_2 is evaluated, local terms are componentwise with |S~| <= |S| + ||E||_F.',
    'Outputs: per head and chunk (or token), ||o~ - o||_F <= ||local||_F + ||Q\' - Intra W||_2 ||E||_F.',
    'Gated RMSNorm: weight * x * rsqrt(mean x^2 + 1e-6) * sigmoid(z) within the composed relative rounding.',
    'Every rounding adds 2^-126 for underflow; every bound is scaled by 1 + 2^-28 to cover the float64 truth.',
]

def silu64(x):
    return x / (1 + np.exp(-x))

def sigmoid64(x):
    return 1 / (1 + np.exp(-x))

def softplus64(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))

def prod(a, da, b, db):
    return da * np.abs(b) + np.abs(a) * db + da * db + U * (np.abs(a) + da) * (np.abs(b) + db) + TINY

def mm(a, da, b, db, n):
    """Bound for a @ b with inner length n (batched over leading axes)."""
    return da @ np.abs(b) + np.abs(a) @ db + da @ db + gamma(n) * ((np.abs(a) + da) @ (np.abs(b) + db)) + n * TINY

def l2n(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)

def ratio(diff, bound):
    return float((diff / bound).max())

def solve_bound(T_, dT, B, dB):
    """Unit lower triangular solve T x = B with perturbed T and B (batched)."""
    C = T_.shape[-1]
    Ti = np.linalg.inv(T_)
    X = Ti @ B
    Fm = dT + gamma(C) * (np.abs(T_) + dT)
    G = np.abs(Ti) @ Fm
    a = np.abs(Ti) @ (dB + Fm @ np.abs(X))
    norm = G.sum(-1).max(-1)
    assert (norm < 1).all(), 'triangular solve bound requires ||G||_inf < 1'
    e = a + (G.sum(-1, keepdims=True)) * (a.max(-2, keepdims=True) / (1 - norm)[..., None, None])
    return X, e + 2.0 ** -30 * (np.abs(Ti) @ np.abs(B))

def exact_delta(q, k, v, g, beta, S0, normalize=True):
    """Independent float64 gated delta rule, token by token, from the mathematical definition."""
    qn, kn = (l2n(q) / math.sqrt(DK), l2n(k)) if normalize else (q / math.sqrt(DK), k)
    S = S0.copy()
    out = np.zeros_like(v)
    for t in range(q.shape[0]):
        S = S * np.exp(g[t])[:, None, None]
        delta = (v[t] - np.einsum('hij,hi->hj', S, kn[t])) * beta[t][:, None]
        S = S + kn[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum('hij,hi->hj', S, qn[t])
    return out, S

def chunk_bound(q, k, v, g, beta, S0, C=64):
    """Derived bound for torch_chunk_gated_delta_rule; returns exact algebra outputs, per-(head, chunk) bounds, final state bound."""
    n_tok = q.shape[0]
    pad = (C - n_tok % C) % C
    to_heads = lambda x: np.moveaxis(np.concatenate([x, np.zeros((pad, *x.shape[1:]))]), 0, 1)
    kn = l2n(k)
    qs = l2n(q) * DK ** -0.5
    K_, Q_, V_ = (to_heads(x).reshape(HV, -1, C, DK) for x in (kn, qs, v))
    dK, dQ = EPS_L2 * np.abs(K_) + TINY, EPS_Q_CHUNK * np.abs(Q_) + TINY
    G_, B_ = (to_heads(x).reshape(HV, -1, C) for x in (g, beta))
    n = G_.shape[1]
    VB = V_ * B_[..., None]
    dVB = U * np.abs(VB) + TINY
    KB = K_ * B_[..., None]
    dKB = prod(K_, dK, B_[..., None], 0)
    cum = np.cumsum(G_, -1)
    dcum = np.array([gamma(t) for t in range(C)]) * np.cumsum(np.abs(G_), -1) + np.arange(1, C + 1) * TINY
    diff = cum[..., :, None] - cum[..., None, :]
    ddiff = dcum[..., :, None] + dcum[..., None, :]
    ddiff = ddiff + U * (np.abs(diff) + ddiff) + TINY
    lower = np.tril(np.ones((C, C), dtype=bool))
    P = np.where(lower, np.exp(np.where(lower, diff, 0)), 0)
    dP = np.where(lower, P * (np.exp(ddiff) * (1 + C_EXP * U) - 1) + TINY, 0)
    Kt, dKt = np.swapaxes(K_, -1, -2), np.swapaxes(dK, -1, -2)
    KK = KB @ Kt
    A_ = KK * P
    dA = np.where(lower, prod(KK, mm(KB, dKB, Kt, dKt, DK), P, dP), 0)
    QK = Q_ @ Kt
    Intra = QK * P
    dIntra = np.where(lower, prod(QK, mm(Q_, dQ, Kt, dKt, DK), P, dP), 0)
    Ec = np.exp(cum)
    dEc = Ec * (np.exp(dcum) * (1 + C_EXP * U) - 1) + TINY
    KBE = KB * Ec[..., None]
    dKBE = prod(KB, dKB, Ec[..., None], dEc[..., None])
    strict = np.tril(np.ones((C, C), dtype=bool), -1)
    T_ = np.eye(C) + np.where(strict, A_, 0)
    dT = np.where(strict, dA, 0)
    NV, dNV = solve_bound(T_, dT, VB, dVB)
    W_, dW = solve_bound(T_, dT, KBE, dKBE)
    Qp, dQp = Q_ * Ec[..., None], prod(Q_, dQ, Ec[..., None], dEc[..., None])
    rest = cum[..., -1:] - cum
    drest = dcum[..., -1:] + dcum
    drest = drest + U * (np.abs(rest) + drest) + TINY
    Ek = np.exp(rest)
    dEk = Ek * (np.exp(drest) * (1 + C_EXP * U) - 1) + TINY
    Kp, dKp = K_ * Ek[..., None], prod(K_, dK, Ek[..., None], dEk[..., None])
    D = np.exp(cum[..., -1])
    dD = D * (np.exp(dcum[..., -1]) * (1 + C_EXP * U) - 1) + TINY
    S = S0.copy()
    eps = np.zeros(HV)
    outs, out_bounds = [], []
    for i in range(n):
        absS = np.abs(S) + eps[:, None, None]
        Wi, dWi = W_[:, i], dW[:, i]
        WS = Wi @ S
        dWS = dWi @ absS + gamma(DK) * ((np.abs(Wi) + dWi) @ absS) + DK * TINY   # local; W E is propagated
        vnew = NV[:, i] - WS
        dv = dNV[:, i] + dWS + U * (np.abs(NV[:, i]) + dNV[:, i] + np.abs(Wi) @ absS + dWS) + TINY
        Weps = np.abs(Wi).sum(-1)[..., None] * eps[:, None, None]
        absV = np.abs(vnew) + dv + Weps
        inter = Qp[:, i] @ S
        dinter = dQp[:, i] @ absS + gamma(DK) * ((np.abs(Qp[:, i]) + dQp[:, i]) @ absS) + DK * TINY
        iv = Intra[:, i] @ vnew
        div = dIntra[:, i] @ absV + np.abs(Intra[:, i]) @ dv + gamma(C) * ((np.abs(Intra[:, i]) + dIntra[:, i]) @ absV) + C * TINY
        Mo = Qp[:, i] - Intra[:, i] @ Wi
        prop_o = np.abs(Mo).sum(-1)[..., None] * eps[:, None, None]
        do = dinter + div + U * (np.abs(inter) + np.abs(iv) + dinter + div + prop_o) + TINY
        out_bounds.append(np.sqrt((do ** 2).sum((-1, -2))) + np.linalg.norm(Mo, 2, axis=(-2, -1)) * eps)
        outs.append(inter + iv)
        Di, dDi = D[:, i][:, None, None], dD[:, i][:, None, None]
        SD = S * Di
        dSD = absS * dDi + U * absS * (Di + dDi) + TINY
        Kpt, dKpt = np.swapaxes(Kp[:, i], -1, -2), np.swapaxes(dKp[:, i], -1, -2)
        KV = Kpt @ vnew
        dKV = dKpt @ absV + np.abs(Kpt) @ dv + gamma(C) * ((np.abs(Kpt) + dKpt) @ absV) + C * TINY
        Ms = Di * np.eye(DK) - Kpt @ Wi
        prop_s = (Di + np.abs(Kpt) @ np.abs(Wi)).sum(-1)[..., None] * eps[:, None, None]
        dS = dSD + dKV + U * (np.abs(SD) + np.abs(KV) + dSD + dKV + prop_s) + TINY
        eps = np.sqrt((dS ** 2).sum((-1, -2))) + np.linalg.norm(Ms, 2, axis=(-2, -1)) * eps
        S = SD + KV
    out = np.moveaxis(np.concatenate(outs, 1), 0, 1)[:n_tok]
    return out, np.stack(out_bounds, 1) * UP, S, eps * UP

def recurrent_bound(q, k, v, g, beta, S0):
    """Derived bound for torch_recurrent_gated_delta_rule; per-(head, token) output bounds and final state bound."""
    kn = l2n(k)
    qs = l2n(q) / math.sqrt(DK)
    dk = EPS_L2 * np.abs(kn) + TINY
    dq = EPS_Q_RECURRENT * np.abs(qs) + TINY
    S = S0.copy()
    eps = np.zeros(HV)
    out_bounds = []
    for t in range(q.shape[0]):
        d = np.exp(g[t])[:, None, None]
        dd = C_EXP * U * d + TINY
        absS = np.abs(S) + eps[:, None, None]
        S1 = S * d
        dS1 = absS * dd + U * absS * (d + dd) + TINY
        absS1 = np.abs(S1) + dS1 + d * eps[:, None, None]
        k_t, dk_t, b_t = kn[t], dk[t], beta[t][:, None]
        kv = np.einsum('hij,hi->hj', S1, k_t)
        dkv = (np.einsum('hij,hi->hj', dS1, np.abs(k_t)) + np.einsum('hij,hi->hj', absS1, dk_t)
               + gamma(DK + 1) * np.einsum('hij,hi->hj', absS1, np.abs(k_t) + dk_t) + DK * TINY)
        kv_prop = d[:, :, 0] * np.abs(k_t).sum(-1, keepdims=True) * eps[:, None]
        delta = (v[t] - kv) * b_t
        ddelta = b_t * dkv + b_t * U * (np.abs(v[t]) + np.abs(kv) + dkv + kv_prop) * (2 + U) + TINY
        absD = np.abs(delta) + ddelta + b_t * kv_prop
        outer = (np.abs(k_t) + dk_t)[:, :, None] * absD[:, None, :]
        dS2 = (dS1 + dk_t[:, :, None] * absD[:, None, :] + np.abs(k_t)[:, :, None] * ddelta[:, None, :] + U * outer
               + U * (np.abs(S1) + dS1 + d * eps[:, None, None] + outer * (1 + U)) + TINY)
        S2 = S1 + k_t[:, :, None] * delta[:, None, :]
        Mnorm = d[:, 0, 0] * np.maximum(1, np.abs(1 - beta[t] * (k_t * k_t).sum(-1)))
        eps = np.sqrt((dS2 ** 2).sum((-1, -2))) + Mnorm * eps
        absS2 = np.abs(S2) + eps[:, None, None]
        local = (np.einsum('hij,hi->hj', absS2, dq[t]) + gamma(DK + 1) * np.einsum('hij,hi->hj', absS2, np.abs(qs[t]) + dq[t]) + DK * TINY)
        out_bounds.append(np.sqrt((local ** 2).sum(-1)) + np.sqrt((qs[t] ** 2).sum(-1)) * eps)
        S = S2
    return np.stack(out_bounds, 1) * UP, S, eps * UP

def f64(t):
    return t.detach().double().numpy()

def linear_check(record, name, weight):
    x, y = (f64(a) for a in record[name])
    W = f64(weight)
    bound = (gamma(W.shape[1]) * (np.abs(x) @ np.abs(W).T) + W.shape[1] * TINY) * UP
    return ratio(np.abs(y - x @ W.T), bound)

conv_w = f64(gdn.conv1d.weight[:, 0])
A64, dt64, normw = f64(gdn.A_log), f64(gdn.dt_bias), f64(gdn.norm.weight)
stage = {name: 0.0 for name in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'conv', 'beta', 'g', 'coreOutput', 'coreState', 'norm', 'out_proj']}
alt_hits = {name: 0 for name in ['norm-gate-silu', 'norm-weight-one-plus', 'no-qk-l2norm', 'heads-tiled-not-interleaved', 'beta-ignored', 'decay-ignored']}
history = {}
semantics = {'convStateIsLastFourRawColumns': True, 'continuationConvInputIsStatePlusNew': True,
             'initialStateIsPreviousFinalState': True, 'chunkAlgebraMatchesRecurrence': 0.0}
for ci, c in enumerate(calls):
    previous = calls[ci - 1] if ci > 0 and calls[ci - 1]['path'] == c['path'] else None
    for name, module in [('in_proj_qkv', gdn.in_proj_qkv), ('in_proj_z', gdn.in_proj_z), ('in_proj_b', gdn.in_proj_b),
                         ('in_proj_a', gdn.in_proj_a), ('out_proj', gdn.out_proj)]:
        stage[name] = max(stage[name], linear_check(c, name, module.weight))
    raw = c['in_proj_qkv'][1].T                                   # (conv_dim, tokens) pre-conv
    kind, conv_in, conv_state, conv_out = c['conv']
    x = f64(conv_in[0])
    if kind == 'fn':
        xp = np.concatenate([np.zeros((x.shape[0], KS - 1)), x], 1)
        windows = np.stack([xp[:, j:j + x.shape[1]] for j in range(KS)], -1)
    else:
        xs = np.concatenate([f64(conv_state[0]), x], 1)
        windows = xs[:, None, -KS:]
    pre = (windows * conv_w[:, None, :]).sum(-1)
    dpre = gamma(KS) * (np.abs(windows) * np.abs(conv_w)[:, None, :]).sum(-1) + KS * TINY
    conv_bound = (L_SILU * dpre + EPS_SILU * (np.abs(silu64(pre)) + L_SILU * dpre) + TINY) * UP
    stage['conv'] = max(stage['conv'], ratio(np.abs(f64(conv_out[0]) - silu64(pre)), conv_bound))
    path_history = history.setdefault(c['path'], [])
    if c['path'] != 'fullSequence':
        if kind == 'fn' and path_history:
            expected_in = torch.cat([previous['convState'][0], raw], -1)
            semantics['continuationConvInputIsStatePlusNew'] &= torch.equal(conv_in[0], expected_in)
        path_history.append(raw)
        full = torch.cat(path_history, -1)
        expected_state = F.pad(full, (max(0, KS - full.shape[-1]), 0))[:, -KS:]
        semantics['convStateIsLastFourRawColumns'] &= torch.equal(c['convState'][0], expected_state)
    core = c['core']
    q, k, v, g, beta = (f64(t[0]) for t in core['inputs'])
    a_in, b_in = f64(c['in_proj_a'][1]), f64(c['in_proj_b'][1])
    beta_exact = sigmoid64(b_in)
    stage['beta'] = max(stage['beta'], ratio(np.abs(beta - beta_exact), (EPS_SIGMOID * beta_exact + TINY) * UP))
    xg = a_in + dt64
    dx = U * np.abs(xg)
    sp = softplus64(xg)
    dsp = dx + L(C_EXP * U) * (1 + C_LOG * U) + C_LOG * U * (sp + dx) + TINY + np.where(np.abs(xg - 20) <= dx, math.exp(-20), 0)
    Aexp = np.exp(A64)
    g_exact = -Aexp * sp
    stage['g'] = max(stage['g'], ratio(np.abs(g - g_exact), prod(Aexp, C_EXP * U * Aexp + TINY, sp, dsp) * UP))
    S0 = np.zeros((HV, DK, DV)) if core['initial'] is None else f64(core['initial'][0])
    if previous is not None and c['path'] != 'fullSequence':
        semantics['initialStateIsPreviousFinalState'] &= core['initial'] is not None and torch.equal(core['initial'], previous['recurrentState'])
    out_exact, S_exact = exact_delta(q, k, v, g, beta, S0)
    computed = f64(core['output'][0])
    if core['kind'] == 'torch_chunk_gated_delta_rule':
        algebra, out_b, S_alg, eps_final = chunk_bound(q, k, v, g, beta, S0)
        semantics['chunkAlgebraMatchesRecurrence'] = max(semantics['chunkAlgebraMatchesRecurrence'],
                                                         float(np.abs(algebra - out_exact).max() / np.abs(out_exact).max()))
        blocks = np.arange(q.shape[0]) // 64
        err = np.zeros_like(out_b)
        np.add.at(err.T, blocks, ((computed - out_exact) ** 2).sum(-1))
        err = np.sqrt(err)
    else:
        out_b, S_rec, eps_final = recurrent_bound(q, k, v, g, beta, S0)
        err = np.sqrt(((computed - out_exact) ** 2).sum(-1)).T
    stage['coreOutput'] = max(stage['coreOutput'], ratio(err, out_b))
    if core['final'] is not None:
        state_err = np.sqrt(((f64(core['final'][0]) - S_exact) ** 2).sum((-1, -2)))
        stage['coreState'] = max(stage['coreState'], ratio(state_err, eps_final))
    # Alternative core semantics: distinguished when some block differs from truth by more than twice its bound.
    q_tiled = np.tile(q[:, ::HV // HK], (1, HV // HK, 1))
    k_tiled = np.tile(k[:, ::HV // HK], (1, HV // HK, 1))
    for name, alt in [('no-qk-l2norm', lambda: exact_delta(q, k, v, g, beta, S0, normalize=False)),
                      ('heads-tiled-not-interleaved', lambda: exact_delta(q_tiled, k_tiled, v, g, beta, S0)),
                      ('beta-ignored', lambda: exact_delta(q, k, v, g, np.ones_like(beta), S0)),
                      ('decay-ignored', lambda: exact_delta(q, k, v, np.zeros_like(g), beta, S0))]:
        alt_out = alt()[0]
        if core['kind'] == 'torch_chunk_gated_delta_rule':
            alt_err = np.zeros_like(out_b)
            np.add.at(alt_err.T, blocks, ((alt_out - out_exact) ** 2).sum(-1))
            alt_err = np.sqrt(alt_err)
        else:
            alt_err = np.sqrt(((alt_out - out_exact) ** 2).sum(-1)).T
        alt_hits[name] += int((alt_err > 2 * out_b).sum())
    xn, zn, yn = (f64(t) for t in c['norm'])
    rms = xn / np.sqrt((xn * xn).mean(-1, keepdims=True) + 1e-6)
    y_exact = normw * rms * sigmoid64(zn)
    norm_bound = (EPS_NORM * np.abs(y_exact) + TINY) * UP
    stage['norm'] = max(stage['norm'], ratio(np.abs(yn - y_exact), norm_bound))
    alt_hits['norm-gate-silu'] += int((np.abs(normw * rms * silu64(zn) - y_exact) > 2 * norm_bound).sum())
    alt_hits['norm-weight-one-plus'] += int((np.abs((1 + normw) * rms * sigmoid64(zn) - y_exact) > 2 * norm_bound).sum())

# ---- End-to-end float64 pipeline from hidden states (observation, not a bound) -------------
def pipeline64(h):
    W = lambda m: f64(m.weight)
    mixed = h @ W(gdn.in_proj_qkv).T
    xp = np.concatenate([np.zeros((KS - 1, mixed.shape[1])), mixed])
    conv = silu64(sum(xp[j:j + h.shape[0]] * conv_w[:, j] for j in range(KS)))
    q, k, v = np.split(conv, [HK * DK, 2 * HK * DK], -1)
    q = np.repeat(q.reshape(-1, HK, DK), HV // HK, 1)
    k = np.repeat(k.reshape(-1, HK, DK), HV // HK, 1)
    v = v.reshape(-1, HV, DV)
    beta = sigmoid64(h @ W(gdn.in_proj_b).T)
    g = -np.exp(A64) * softplus64(h @ W(gdn.in_proj_a).T + dt64)
    core, S = exact_delta(q, k, v, g, beta, np.zeros((HV, DK, DV)))
    z = (h @ W(gdn.in_proj_z).T).reshape(-1, HV, DV)
    y = normw * core / np.sqrt((core * core).mean(-1, keepdims=True) + 1e-6) * sigmoid64(z)
    return y.reshape(h.shape[0], -1) @ W(gdn.out_proj).T, S
truth64, final64 = pipeline64(f64(hidden[0]))
end_to_end = {path: {'maxAbsError': float(np.abs(f64(out) - truth64).max()),
                     'maxRelativeError': float(np.abs(f64(out) - truth64).max() / np.abs(truth64).max())}
              for path, out in outputs.items()}
for path in ['chunkedPrefillThenDecode', 'shortPrefillThenDecode']:
    final = next(c for c in reversed(calls) if c['path'] == path)['recurrentState'][0]
    end_to_end[path]['finalStateRelativeError'] = float(np.linalg.norm(f64(final) - final64) / np.linalg.norm(final64))
cross_path = {f'{a}|{b}': float((outputs[a] - outputs[b]).abs().max()) for a, b in
              [('fullSequence', 'chunkedPrefillThenDecode'), ('fullSequence', 'shortPrefillThenDecode')]}

rejections = {}
for label, call in [('hiddenWidth2559', lambda: gdn(hidden[:, :4, :-1])),
                    ('cacheLayerNotLinearAttention', lambda: gdn(hidden[:, :4], cache_params=CachePort([object()])))]:
    try:
        with torch.no_grad():
            call()
        rejections[label] = {'rejected': False}
    except (RuntimeError, ValueError, IndexError) as error:
        rejections[label] = {'rejected': True, 'error': type(error).__name__}

alternatives = {name: {'blocksBeyondTwiceBound': hits, 'distinguished': hits > 0} for name, hits in alt_hits.items()}
checks = {f'{name}WithinDerivedBound': value <= 1 for name, value in stage.items()}
checks.update({key: bool(value) for key, value in semantics.items() if key != 'chunkAlgebraMatchesRecurrence'})
checks['chunkAlgebraMatchesRecurrence'] = semantics['chunkAlgebraMatchesRecurrence'] < 1e-9
checks['malformedInputsRejected'] = all(r['rejected'] for r in rejections.values())
checks['alternativeSemanticsDistinguished'] = all(a['distinguished'] for a in alternatives.values())
report = {
    'schema': 'doppler.qwen38-linear-attention-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU',
    'environment': {'machine': platform.machine(), 'blas': (re.search(r'BLAS_INFO=([^,\s]+)', torch.__config__.show()) or [None, None])[1],
                    'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'threads': torch.get_num_threads()},
    'scope': 'Unchanged upstream Qwen4ExpTextGatedDeltaNet, conv and gated-delta torch bodies, gated RMSNorm and the pinned '
             'LinearAttentionLayer cache at the pinned text geometry with synthetic F32 weights; no checkpoint weights or Doppler execution',
    'sourceNodes': {'modeling_qwen4_exp.py': GDN_NAMES, 'cache_utils.py': ['LinearAttentionCacheLayerMixin', 'LinearAttentionLayer',
                    'Cache.update_conv_state', 'Cache.update_recurrent_state', 'Cache.has_previous_state'], 'activations.py': ['SiLUActivation']},
    'testPorts': ['CachePort holds pinned LinearAttentionLayer objects and the three unchanged Cache methods',
                  'stage spies record inputs and outputs and return the upstream results unchanged',
                  'fla and causal_conv1d kernels are CUDA/Triton and not installed; the export-only forward-substitution branch is not run'],
    'geometry': {'layer': LAYER, 'linearLayers': cfg.layer_types.count('linear_attention'), 'hiddenSize': HID, 'keyHeads': HK,
                 'valueHeads': HV, 'keyHeadDim': DK, 'valueHeadDim': DV, 'convDim': gdn.conv_dim, 'convKernel': KS,
                 'chunkSize': 64, 'normGate': gdn.norm.activation, 'tokens': T},
    'recipe': {'values': 'splitmix64(index + stream<<40) >> 40, minus 2^23, times 2^(exponent-23); exact F32', 'weights': weights,
               'hidden': {'stream': 4096, 'exponent': 1, 'sha256': digest(hidden)}},
    'paths': {path: {'calls': len(spans), 'spans': spans[:3] + (['...'] if len(spans) > 3 else []),
                     'outputSha256': digest(outputs[path])} for path, spans in PATHS.items()},
    'derivation': DERIVATION,
    'constants': {'u': U, 'epsL2': EPS_L2, 'epsQueryChunk': EPS_Q_CHUNK, 'epsQueryRecurrent': EPS_Q_RECURRENT, 'epsNorm': EPS_NORM,
                  'epsSigmoid': EPS_SIGMOID, 'lSilu': L_SILU, 'cExp': C_EXP, 'cLog1p': C_LOG},
    'stageMaxRatio': stage,
    'semantics': semantics,
    'endToEndObservation': {'note': 'float64 pipeline from hidden states; observation only, the bounds are stage-wise',
                            'paths': end_to_end, 'maxCrossPathDifference': cross_path},
    'alternativeSemantics': alternatives,
    'qwen35Comparison': qwen35_comparison,
    'malformedRejected': rejections,
    'notEstablished': ['Learned weights, bfloat16 checkpoint dtype and the A_log/dt_bias distribution', 'FLA chunk and fused recurrent kernels '
                       'and causal_conv1d CUDA kernels (GPU only)', 'Padding masks (apply_mask_to_padding_states) and record_past/crop rollback',
                       'The export-only forward-substitution solve', 'Doppler gated-delta kernels'],
    'checks': checks,
}
report['passed'] = all(checks.values())
report['modelQualified'] = False
(ROOT / 'linear-attention-reference.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks, 'stage': stage, 'semantics': semantics,
                  'alternatives': alternatives, 'endToEnd': end_to_end, 'crossPath': cross_path,
                  'qwen35': {k: v['identical'] for k, v in qwen35_comparison.items()}}))
if not report['passed']:
    raise SystemExit(1)
