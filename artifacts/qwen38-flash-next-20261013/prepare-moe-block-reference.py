"""Pinned upstream MoE block at full Flash-Next geometry; synthetic CPU fixtures, not compatibility.

512 experts, top-10 routing, 640-wide experts, shared expert and sigmoid shared gate.
The unchanged eager expert body runs through the pinned use_experts_implementation
dispatcher, which also selects the pinned batched_mm and grouped_mm implementations.
Every output is checked against float64 truth with a derived F32 bound.
Run: python3 -I artifacts/qwen38-flash-next-20261013/prepare-moe-block-reference.py
"""
import ast
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
config_tree = ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text())
config_class = next(n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
field = next(n for n in config_class.body if isinstance(n, ast.AnnAssign) and n.target.id == 'norm_topk_prob')
config.setdefault('norm_topk_prob', ast.literal_eval(field.value))
cfg = SimpleNamespace(**config)
assert torch.get_float32_matmul_precision() == 'highest'
torch.set_num_threads(2)

def source_nodes(path, names, decorators):
    """Unchanged top-level definitions; listed decorators are asserted and removed explicitly."""
    found = {}
    for node in ast.parse((ROOT / 'source' / path).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            assert [ast.unparse(d) for d in node.decorator_list] == decorators.get(node.name, []), node.name
            if node.name in decorators and decorators[node.name] != ['use_experts_implementation']:
                node.decorator_list = []
            found[node.name] = node
    assert set(found) == set(names), set(names) - set(found)
    return [found[name] for name in names]

def run(nodes, namespace, label):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), label, 'exec'), namespace)

# ACT2FN entries come from the pinned activations mapping (hub-kernel decorator bypassed:
# the `kernels` package is not installed, so upstream also resolves to the torch body).
activations = ast.parse((ROOT / 'source/activations.py').read_text())
act2cls = next(n.value for n in activations.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ACT2CLS')
mapping = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(act2cls.keys, act2cls.values)}
assert mapping['silu'] == 'SiLUActivation' and mapping['sigmoid'] == 'nn.Sigmoid'
namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'wraps': wraps, 'Tensor': torch.Tensor}
run(source_nodes('activations.py', ['SiLUActivation'], {'SiLUActivation': ["use_kernel_forward_from_hub('SiLU')"]}), namespace, '<pinned-activations>')
namespace['ACT2FN'] = {'silu': namespace['SiLUActivation'](), 'sigmoid': nn.Sigmoid()}

def version_tuple(text):
    return tuple(int(part) for part in re.match(r'(\d+)\.(\d+)(?:\.(\d+))?', text).groups(default='0'))

class ExpertsRegistry:
    """Test stub with ExpertsInterface.get_interface semantics for the CPU-runnable entries."""
    def __init__(self):
        self.functions = {}
    def get_interface(self, name, default):
        if name == 'eager':
            return default
        if name not in self.functions:
            raise KeyError(f'{name} is not a registered experts implementation')
        return self.functions[name]

registry = ExpertsRegistry()
namespace.update({
    'logger': SimpleNamespace(warning_once=lambda *a, **k: None),
    'is_torchdynamo_compiling': lambda: False,
    'is_torch_less_or_equal': lambda v, accept_dev=False: version_tuple(torch.__version__) <= version_tuple(v),
    'is_torch_greater_or_equal': lambda v, accept_dev=False: version_tuple(torch.__version__) >= version_tuple(v),
    'ALL_EXPERTS_FUNCTIONS': registry, 'ExpertsInterface': ExpertsRegistry,
})
moe_names = ['_batched_linear', 'batched_mm_experts_forward', '_can_use_grouped_mm', '_grouped_mm', '_grouped_linear',
             'grouped_mm_experts_forward', '_default_apply_gate', 'use_experts_implementation']
run(source_nodes('integrations_moe.py', moe_names, {}), namespace, '<pinned-moe-integrations>')
registry.functions.update(batched_mm=namespace['batched_mm_experts_forward'], grouped_mm=namespace['grouped_mm_experts_forward'])
model_names = ['Qwen4ExpTextMLP', 'Qwen4ExpTextExperts', 'Qwen4ExpTextTopKRouter', 'Qwen4ExpTextSparseMoeBlock']
run(source_nodes('modeling_qwen4_exp.py', model_names, {'Qwen4ExpTextExperts': ['use_experts_implementation']}),
    namespace, '<pinned-moe-block>')

# ---- Deterministic dyadic synthetic values -------------------------------------
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

H, E, K, I = cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size
GATE_UP_STREAM, DOWN_STREAM = 1 << 16, 2 << 16

class ExpertWeights:
    """Test port for the 3D expert parameter: synthetic expert e is generated on access.

    The dense F32 tensors would be 6.7 GB and 3.4 GB; the eager and batched_mm bodies
    only index experts, so they run unchanged against this port."""
    def __init__(self, per_expert, stream, exponent):
        self.per_expert, self.stream, self.exponent, self.cache = per_expert, stream, exponent, {}
        self.shape, self.dtype, self.device = (E, *per_expert), torch.float32, torch.device('cpu')
    def expert(self, e):
        e = int(e)
        if not 0 <= e < E:
            raise IndexError(f'expert {e} outside [0, {E})')
        if e not in self.cache:
            self.cache[e] = dyadic(self.per_expert, self.stream + e, self.exponent)
        return self.cache[e]
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor) and index.dim() == 1:
            return torch.stack([self.expert(i) for i in index.tolist()])
        return self.expert(index)

cfg._experts_implementation = 'eager'
with torch.device('meta'):
    block = namespace['Qwen4ExpTextSparseMoeBlock'](cfg)
weights = []
with torch.no_grad():
    for stream, (name, module) in enumerate([('gate', block.gate), ('shared_expert', block.shared_expert),
                                             ('shared_expert_gate', block.shared_expert_gate)]):
        module.to_empty(device='cpu')
        for ordinal, (pname, parameter) in enumerate(sorted(module.named_parameters())):
            exponent = -4 if pname.startswith('down_proj') else -5
            parameter.copy_(dyadic(tuple(parameter.shape), 64 * (stream + 1) + ordinal, exponent))
            weights.append({'name': f'{name}.{pname}', 'shape': list(parameter.shape), 'stream': 64 * (stream + 1) + ordinal,
                            'exponent': exponent, 'sha256': digest(parameter)})
experts = block.experts
del experts.gate_up_proj, experts.down_proj
experts.gate_up_proj = ExpertWeights((2 * I, H), GATE_UP_STREAM, -5)
experts.down_proj = ExpertWeights((H, I), DOWN_STREAM, -4)
weights += [{'name': 'experts.gate_up_proj[e]', 'shape': [2 * I, H], 'stream': f'{GATE_UP_STREAM} + e', 'exponent': -5},
            {'name': 'experts.down_proj[e]', 'shape': [H, I], 'stream': f'{DOWN_STREAM} + e', 'exponent': -4}]

T = 8
hidden = dyadic((T, H), 4096, 1)
hidden[1] = hidden[0]
hidden[1, :64] = dyadic((64,), 4097, 1)       # near-duplicate: shares most experts with token 0
hidden[7] = hidden[6]                          # exact duplicate: identical routing
captured = {}
def keep(name):
    def hook(module, args, output):
        captured[name] = ([a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args],
                          tuple(o.detach().clone() for o in output) if isinstance(output, tuple) else output.detach().clone())
    return hook
for name, module in [('gate', block.gate), ('experts', block.experts), ('shared_expert', block.shared_expert),
                     ('shared_gate_proj', block.shared_expert.gate_proj), ('shared_up_proj', block.shared_expert.up_proj),
                     ('shared_down_proj', block.shared_expert.down_proj), ('shared_expert_gate', block.shared_expert_gate)]:
    module.register_forward_hook(keep(name))
with torch.no_grad():
    block_output = block(hidden.view(1, T, H)).view(T, H)
logits, routing, indices = captured['gate'][1]
expert_output = captured['experts'][1]

# ---- Derived F32 bounds -------------------------------------------------------------
U, C_EXP, TINY = 2.0 ** -24, 4, 2.0 ** -126
# Each float64 truth rounding term is the F32 term with u replaced by 2^-53 (ratio 2^-29),
# so scaling every bound by 1 + 2^-28 also covers the float64 truth and the bound's own evaluation.
UP = 1 + 2.0 ** -28
gamma = lambda n: n * U / (1 - n * U)
DENOM = (1 + C_EXP * U) * (1 + U) - 1           # relative error of fl(1 + fl(exp(-x)))
EPS_SIGMOID = (1 + U) / (1 - DENOM) - 1         # 1 / (1 + exp(-x)): reciprocal after the sum
EPS_SILU = EPS_SIGMOID                          # x / (1 + exp(-x)): one division after the sum
L_SILU, L_SIGMOID = 1.0999, 0.25                # max |silu'| = 1.09984 at x = 2.3994; max sigmoid' = 1/4
grid = np.linspace(-40, 40, 800001)
sig = 1 / (1 + np.exp(-grid))
assert (sig * (1 + grid * (1 - sig))).max() < L_SILU
DERIVATION = [
    'Linear: |fl(Wx) - Wx| <= gamma_n |W||x| for any summation order (n = 2560 or 640), float32 matmul precision "highest".',
    'PyTorch CPU float kernels at the executed commit: silu = x / (1 + exp(-x)), sigmoid = 1 / (1 + exp(-x)); exp within 4u (2 ulp).',
    f'Activation of a perturbed argument: |f(x~) - f(x)| <= L|dx| with L = {L_SILU} (silu) or 1/4 (sigmoid), plus relative rounding '
    f'{EPS_SILU:.3e} of the computed value.',
    'Products and sums carry |da||b| + |a||db| + |da||db| + u(|a|+|da|)(|b|+|db|) and da + db + u(|a+b| + da + db); every rounding adds 2^-126 for underflow.',
    'Expert combine: each token sums its 10 weighted expert outputs (index_add_ into zeros, or view(T, 10, H).sum(1)); any order gives gamma_9.',
    'The same order-independent bound covers the eager, batched_mm and grouped_mm implementations.',
    'Routing weights and indices are the captured router outputs; the router has its own derived bound in expert-reference-fixtures.json.',
    'Float64 truth: every bound is scaled by 1 + 2^-28, which covers float64 rounding (2^-29 of each F32 term).',
]

def linear_bound(weight, x, dx=None):
    absw = np.abs(weight)
    dx = np.zeros_like(x) if dx is None else dx
    n = weight.shape[1]
    return absw @ dx + gamma(n) * (absw @ (np.abs(x) + dx)) + n * TINY

def silu64(x):
    return x / (1 + np.exp(-x))

def silu_bound(x, dx):
    return L_SILU * dx + EPS_SILU * (np.abs(silu64(x)) + L_SILU * dx) + TINY

def product_bound(a, da, b, db):
    return da * np.abs(b) + np.abs(a) * db + da * db + U * (np.abs(a) + da) * (np.abs(b) + db) + TINY

def gated_mlp(x, gate_up, down):
    """Exact gated MLP and its bound for one input row; gate_up rows are [gate; up]."""
    g = gate_up @ x
    dg = linear_bound(gate_up, x)
    gate, up, dgate, dup = g[:I], g[I:], dg[:I], dg[I:]
    a, da = silu64(gate), silu_bound(gate, dgate)
    h, dh = a * up, product_bound(a, da, up, dup)
    return down @ h, linear_bound(down, h, dh), down @ (silu64(up) * gate)

x64 = hidden.double().numpy()
idx = indices.numpy()
w64 = routing.double().numpy()
truth = np.zeros((T, H))
bound = np.zeros((T, H))
abs_sum = np.zeros((T, H))
pair_out, pair_swapped = np.zeros((T, K, H)), np.zeros((T, K, H))
for t in range(T):
    for k in range(K):
        e = int(idx[t, k])
        y, dy, pair_swapped[t, k] = gated_mlp(x64[t], experts.gate_up_proj.expert(e).double().numpy(),
                                              experts.down_proj.expert(e).double().numpy())
        pair_out[t, k] = y
        truth[t] += w64[t, k] * y
        dyw = abs(w64[t, k]) * dy + U * abs(w64[t, k]) * (np.abs(y) + dy) + TINY
        bound[t] += dyw
        abs_sum[t] += abs(w64[t, k]) * np.abs(y) + dyw
bound = (bound + gamma(K - 1) * abs_sum + K * TINY) * UP

def ratio(actual, reference, limit):
    return float((np.abs(actual.double().numpy() - reference) / limit).max())

results = {'eager': expert_output}
cfg._experts_implementation = 'batched_mm'
with torch.no_grad():
    results['batched_mm'] = experts(hidden, indices, routing)
# grouped_mm keeps all experts as one dense tensor; run it on the hit experts only,
# with a monotonic id remap so grouping and order are unchanged.
hit = sorted(set(idx.reshape(-1).tolist()))
compact_cfg = SimpleNamespace(**{**vars(cfg), 'num_experts': len(hit), '_experts_implementation': 'grouped_mm'})
compact = namespace['Qwen4ExpTextExperts'](compact_cfg)
with torch.no_grad():
    compact.gate_up_proj.copy_(torch.stack([experts.gate_up_proj.expert(e) for e in hit]))
    compact.down_proj.copy_(torch.stack([experts.down_proj.expert(e) for e in hit]))
    remapped = torch.tensor(np.searchsorted(hit, idx), dtype=indices.dtype)
    grouped_kernel = namespace['_can_use_grouped_mm'](hidden, compact.gate_up_proj.transpose(-2, -1), torch.zeros(len(hit), dtype=torch.int32))
    results['grouped_mm'] = compact(hidden, remapped, routing)
implementations = {name: {'maxRatio': ratio(out, truth, bound),
                          'maxAbsDifferenceFromEager': float((out - expert_output).abs().max()),
                          'bitwiseEqualToEager': bool(torch.equal(out, expert_output)), 'sha256': digest(out)}
                   for name, out in results.items()}
implementations['grouped_mm']['torchGroupedMmKernel'] = bool(grouped_kernel)
implementations['grouped_mm']['compactedExperts'] = len(hit)

# Shared expert, its sigmoid gate and the block sum.
sm = block.shared_expert
shared_gate_up = torch.cat([sm.gate_proj.weight, sm.up_proj.weight]).detach().double().numpy()
shared_down = sm.down_proj.weight.detach().double().numpy()
shared_gate_w = block.shared_expert_gate.weight.detach().double().numpy()
shared_truth, shared_bound, block_truth, block_bound, gate_logit = [], [], [], [], []
for t in range(T):
    m, dm, _ = gated_mlp(x64[t], shared_gate_up, shared_down)
    z = float(shared_gate_w[0] @ x64[t])
    gate_logit.append(z)
    dz = float(linear_bound(shared_gate_w, x64[t])[0])
    s = 1 / (1 + math.exp(-z))
    ds = L_SIGMOID * dz + EPS_SIGMOID * (s + L_SIGMOID * dz) + TINY
    sm_t, dsm = s * m, product_bound(s, ds, m, dm)
    shared_truth.append(m)
    shared_bound.append(dm * UP)
    total = truth[t] + sm_t
    block_truth.append(total)
    block_bound.append((bound[t] + dsm + U * (np.abs(truth[t]) + np.abs(sm_t) + bound[t] + dsm) + TINY) * UP)
shared_ratio = ratio(captured['shared_expert'][1], np.array(shared_truth), np.array(shared_bound))
block_ratio = ratio(block_output, np.array(block_truth), np.array(block_bound))

# Router logits at this geometry (selection stability is reported, not assumed).
logit_truth = x64 @ block.gate.weight.detach().double().numpy().T
logit_bound = (gamma(H) * (np.abs(x64) @ np.abs(block.gate.weight.detach().double().numpy()).T) + H * TINY) * UP
logit_ratio = ratio(logits, logit_truth, logit_bound)
order = np.sort(logit_truth, axis=1)
selection_margin = (order[:, -K] - order[:, -K - 1]) - 2 * logit_bound.max(axis=1) - 2 * (math.log1p(C_EXP * U) + 2 * math.log1p(U))

# Alternative semantics: an implementation of one of these differs from truth by more than
# twice the derived bound somewhere, so the bound check rejects it.
block_truth, block_bound = np.array(block_truth), np.array(block_bound)
gate_logit, shared_m = np.array(gate_logit)[:, None], np.array(shared_truth)
probabilities = np.exp(logit_truth - logit_truth.max(1, keepdims=True))
probabilities /= probabilities.sum(1, keepdims=True)
unnormalized = np.take_along_axis(probabilities, idx, 1)
routed = lambda weights_, outs: np.einsum('tk,tkh->th', weights_, outs)
sigmoid_gate = 1 / (1 + np.exp(-gate_logit))
alternative_outputs = {
    'shared-gate-silu': routed(w64, pair_out) + gate_logit * sigmoid_gate * shared_m,
    'shared-gate-absent': routed(w64, pair_out) + shared_m,
    'gate-up-swapped': routed(w64, pair_swapped) + sigmoid_gate * shared_m,
    'routing-weights-ignored': routed(np.ones_like(w64), pair_out) + sigmoid_gate * shared_m,
    'topk-not-renormalized': routed(unnormalized, pair_out) + sigmoid_gate * shared_m,
}
assert np.allclose(routed(w64, pair_out) + sigmoid_gate * shared_m, block_truth, rtol=0, atol=1e-12)
alternatives = {name: {'elementsBeyondTwiceBound': int((np.abs(alt - block_truth) > 2 * block_bound).sum()),
                       'maxDistanceOverTwiceBound': float((np.abs(alt - block_truth) / (2 * block_bound)).max())}
                for name, alt in alternative_outputs.items()}
for item in alternatives.values():
    item['distinguished'] = item['elementsBeyondTwiceBound'] > 0

# Malformed inputs.
rejections = {}
for label, call in [('hiddenWidth2559', lambda: block(hidden[:, :-1].reshape(1, T, H - 1))),
                    ('expertIdBeyondSentinel', lambda: experts(hidden, indices.clone().fill_(E + 1), routing))]:
    cfg._experts_implementation = 'eager'
    try:
        with torch.no_grad():
            call()
        rejections[label] = {'rejected': False}
    except (RuntimeError, IndexError, ValueError) as error:
        rejections[label] = {'rejected': True, 'error': type(error).__name__}
with torch.no_grad():
    sentinel_out = experts(hidden[:2], torch.full((2, K), E), routing[:2])
sentinel_ignored = bool((sentinel_out == 0).all())

per_expert_tokens = np.bincount(idx.reshape(-1), minlength=E)
checks = {
    'eagerWithinDerivedBound': implementations['eager']['maxRatio'] <= 1,
    'batchedMmWithinDerivedBound': implementations['batched_mm']['maxRatio'] <= 1,
    'groupedMmWithinDerivedBound': implementations['grouped_mm']['maxRatio'] <= 1,
    'sharedExpertWithinDerivedBound': shared_ratio <= 1,
    'blockOutputWithinDerivedBound': block_ratio <= 1,
    'routerLogitsWithinDerivedBound': logit_ratio <= 1,
    'malformedInputsRejected': all(r['rejected'] for r in rejections.values()),
    'sentinelExpertSlotIgnored': sentinel_ignored,
    'alternativeSemanticsDistinguished': all(a['distinguished'] for a in alternatives.values()),
}
report = {
    'schema': 'doppler.qwen38-moe-block-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU',
    'environment': {'machine': platform.machine(), 'blas': (re.search(r'BLAS_INFO=([^,\s]+)', torch.__config__.show()) or [None, None])[1],
                    'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'threads': torch.get_num_threads()},
    'scope': 'Unchanged upstream SparseMoeBlock, router, eager experts, shared expert and sigmoid gate at full geometry with synthetic F32 '
             'weights; pinned batched_mm and grouped_mm implementations; no checkpoint weights or Doppler execution',
    'sourceNodes': {'modeling_qwen4_exp.py': model_names, 'integrations_moe.py': moe_names, 'activations.py': ['SiLUActivation']},
    'testPorts': ['ExpertWeights generates synthetic expert e on access instead of the dense 512-expert tensor',
                  'ExpertsRegistry stub resolves eager, batched_mm and grouped_mm like ExpertsInterface',
                  'grouped_mm runs on the hit experts with a monotonic id remap because it needs one dense weight tensor',
                  'deepgemm and sonicmoe are CUDA implementations and are not run'],
    'geometry': {'hiddenSize': H, 'experts': E, 'topK': K, 'expertIntermediate': I,
                 'sharedIntermediate': cfg.shared_expert_intermediate_size, 'activation': cfg.hidden_act, 'tokens': T},
    'recipe': {'values': 'splitmix64(index + stream<<40) >> 40, minus 2^23, times 2^(exponent-23); exact F32', 'weights': weights,
               'hidden': {'stream': 4096, 'exponent': 1, 'token1': 'token 0 with elements 0..63 from stream 4097', 'token7': 'copy of token 6',
                          'sha256': digest(hidden)}},
    'routing': {'indices': indices.tolist(), 'weightsSha256': digest(routing), 'distinctExperts': len(hit),
                'maxTokensPerExpert': int(per_expert_tokens.max()), 'expertsShared': int((per_expert_tokens > 1).sum()),
                'logitMaxRatio': logit_ratio, 'selectionProvenStableTokens': int((selection_margin > 0).sum()),
                'minSelectionMargin': float(selection_margin.min())},
    'derivation': DERIVATION,
    'constants': {'u': U, 'gamma2560': gamma(H), 'gamma640': gamma(I), 'gamma9': gamma(K - 1), 'epsSilu': EPS_SILU,
                  'epsSigmoid': EPS_SIGMOID, 'lSilu': L_SILU, 'cExp': C_EXP},
    'experts': {'implementations': implementations, 'maxBound': float(bound.max()), 'minBound': float(bound.min()),
                'maxObservedEager': float(np.abs(expert_output.double().numpy() - truth).max()),
                'duplicateTokenOutputsBitwiseEqual': bool(torch.equal(expert_output[6], expert_output[7]))},
    'sharedExpert': {'maxRatio': shared_ratio},
    'block': {'maxRatio': block_ratio, 'outputSha256': digest(block_output), 'maxBound': float(block_bound.max()),
              'maxObserved': float(np.abs(block_output.double().numpy() - block_truth).max()),
              'maxAbsOutput': float(np.abs(block_truth).max())},
    'alternativeSemantics': alternatives,
    'boundTightness': 'Worst-case gamma_n bounds with cancellation (|W||x| >> |Wx|) and propagation through two projections are '
                      'about 1e5 times the observed error here; they bound rounding rigorously, and alternative semantics are '
                      'checked to exceed twice the bound, but they are not regression tolerances.',
    'malformedRejected': rejections,
    'notEstablished': ['Learned weights and their routing distribution', 'bfloat16 checkpoint dtype (eager index_add_ accumulates in the output dtype)',
                       'deepgemm, sonicmoe and CUDA grouped_mm', 'Expert-parallel sentinel masking', 'Doppler expert execution'],
    'checks': checks,
}
report['passed'] = all(checks.values())
report['modelQualified'] = False
(ROOT / 'moe-block-reference.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks, 'implementations': implementations,
                  'shared': shared_ratio, 'block': block_ratio, 'alternatives': alternatives, 'routing': {k: report['routing'][k] for k in
                  ['distinctExperts', 'maxTokensPerExpert', 'expertsShared', 'selectionProvenStableTokens', 'minSelectionMargin']}}))
if not report['passed']:
    raise SystemExit(1)
