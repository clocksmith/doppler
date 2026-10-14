"""Pinned upstream eager expert controls; synthetic CPU fixtures, not compatibility.

Router checks use a derived forward error bound for the F32 operations the
upstream eager router executes (see ROUTER_DERIVATION), not a chosen tolerance.
"""
import ast
import hashlib
import json
import math
import platform
import re
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
source = ast.parse((ROOT / 'source/modeling_qwen4_exp.py').read_text())
selected = {'Qwen4ExpTextExperts', 'Qwen4ExpTextTopKRouter'}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
nodes.extend(node for node in source.body if isinstance(node, ast.ClassDef) and node.name in selected)
assert len(nodes) == 3
# Deliberately select the retained eager body; no external accelerated decorator.
namespace = {'torch': torch, 'nn': nn, 'F': F, 'ACT2FN': {'silu': F.silu},
             'use_experts_implementation': lambda cls: cls}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-eager-expert-reference>', 'exec'), namespace)
torch.set_num_threads(2)
torch.manual_seed(0)
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
config_source = ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text())
config_class = next(n for n in config_source.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
field = next(n for n in config_class.body if isinstance(n, ast.AnnAssign) and n.target.id == 'norm_topk_prob')
config.setdefault('norm_topk_prob', ast.literal_eval(field.value))

def record(t):
    t = t.detach().cpu().contiguous()
    return {'shape':list(t.shape), 'dtype':str(t.dtype),
            'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest(), 'values':t.tolist()}

# The bound below is valid only for IEEE F32 round-to-nearest arithmetic.
assert torch.get_float32_matmul_precision() == 'highest'
blas = re.search(r'BLAS_INFO=([^,\s]+)', torch.__config__.show())
router = namespace['Qwen4ExpTextTopKRouter'](SimpleNamespace(**config))
base = (torch.arange(config['hidden_size']) % 13 - 6).float() / 128
scale = (torch.arange(config['num_experts']) + 1).float() / 1024
with torch.no_grad():
    router.weight.copy_(scale[:,None] * base[None,:])
    inputs = torch.stack([base, -base])
    logits, probabilities, indices = router(inputs)
assert logits.dtype == probabilities.dtype == router.weight.dtype == torch.float32
H, E, K = config['hidden_size'], config['num_experts'], config['num_experts_per_tok']
expected_indices = [list(range(E-1, E-K-1, -1)), list(range(K))]
assert indices.tolist() == expected_indices
# Products of F32 values are exact in float64; fsum returns the correctly rounded sum.
expected_logits = [[math.fsum(float(a)*float(b) for a,b in zip(row,weight))
                    for weight in router.weight.tolist()] for row in inputs.tolist()]
logit_error = max(abs(float(logits[r,c])-expected_logits[r][c]) for r in range(2) for c in range(E))
expected_scores=[]
for r in range(2):
    retained=[expected_logits[r][i] for i in expected_indices[r]]
    exps=[math.exp(v-max(retained)) for v in retained]
    expected_scores.append([v/math.fsum(exps) for v in exps])
score_error=max(abs(float(probabilities[r,c])-expected_scores[r][c]) for r in range(2) for c in range(K))

# ---- Derived forward error bound for the executed F32 router -------------------
U = 2.0 ** -24                     # F32 unit roundoff, round to nearest
C_EXP = 4                          # F32 exp relative error <= C_EXP*U (2 ulp; Sleef_expf4_u10 documents 1.0 ulp)
UP = 1 + 2.0 ** -40                # outward rounding for float64 evaluation of each bound
TRUTH = 16 * 2.0 ** -53            # relative error allowance for the float64 truth itself
gamma = lambda n: n * U / (1 - n * U)
L = lambda t: -math.log1p(-t)      # |log(1+d)| <= L(t) whenever |d| <= t
ROUTER_DERIVATION = [
    'Upstream order: logits = F.linear(x, W); p = softmax(logits, dtype=float32, dim=-1) over all 512 experts; '
    'v, idx = topk(p, 10); v /= v.sum(-1, keepdim=True); v.to(float32). Sigmoid is not used by this router; it gates only the shared expert.',
    'Logits: |fl(x.w_e) - x.w_e| <= gamma_n * sum_i |x_i w_ei|, n = 2560, gamma_n = n*u/(1-n*u), u = 2^-24, '
    'for any summation order with or without FMA (Higham, Accuracy and Stability, eq. 3.5). Requires float32 matmul precision "highest".',
    'Softmax (PyTorch _vec_softmax_lastdim for float): m = max (exact); E_j = exp(fl(l_j - m)) with fl(l_j - m) = (l_j - m)(1+a), |a| <= u, '
    'and exp relative error <= C_EXP*u; Z = sum E_j and R = fl(1/Z) are common to the row; p_j = fl(E_j * R), one rounding.',
    'Renormalization cancels the common Z and R exactly: v_i = E_i(1+r_i)(1+d_i) / sum_S E_j(1+r_j)(1+t_j), |t_j| <= gamma_{k-1} for the '
    '10-term sum in any order, |d_i| <= u for the final division. In exact arithmetic v_i = exp(l_i)/sum_S exp(l_j), so the exact reference is '
    'softmax over the retained exact logits.',
    'With beta_j the logit bound and L(t) = -log(1-t): |log(v_i/s_i)| <= N_i + D, N_i = beta_i + u|l_i - m| + L(C_EXP u) + 2 L(u), '
    'D = max_S(beta_j + u|l_j - m|) + L(C_EXP u) + L(u) + L(gamma_{k-1}); hence |v_i - s_i| <= s_i * expm1(N_i + D).',
    'Selection is provably the exact top-k when min_S(l_j - beta_j - u|l_j - m|) - max_notS(l_k + beta_k + u|l_k - m|) - 2 L(C_EXP u) - 2 L(u) > 0.',
    'Each bound is multiplied by (1 + 2^-40) for its own float64 evaluation and allows 16 * 2^-53 relative error in the float64 truth.',
]
X64 = inputs.double().numpy()
W64 = router.weight.detach().double().numpy()
P = X64[:, None, :] * W64[None, :, :]                      # exact products, shape (2, E, H)
A = np.array([[math.fsum(abs(v) for v in P[r, e]) for e in range(E)] for r in range(2)])
beta = gamma(H) * A * UP
truth = np.array(expected_logits)
computed = logits.double().numpy()
logit_diff = np.abs(computed - truth)
logit_bound = beta + TRUTH * np.abs(truth)
worst = np.unravel_index(np.argmax(logit_diff), logit_diff.shape)
logit_ratio = logit_diff / logit_bound
logit_passed = bool((logit_diff <= logit_bound).all())

# Structural exactness control: if every product is an F32 integer multiple of a
# power-of-two quantum q and sum|p| < 2^24 q, every subset sum is representable,
# so any summation order (with or without FMA) returns the exact dot product.
assert (P.astype(np.float32).astype(np.float64) == P).all()
nonzero = P != 0
mant, expo = np.frexp(np.where(nonzero, P, 1.0))
ints = np.abs(np.rint(mant * 2.0 ** 53)).astype(np.int64)
valuation = expo.astype(np.int64) - 53 + np.log2(ints & -ints).astype(np.int64)
quantum_exp = np.where(nonzero, valuation, np.iinfo(np.int64).max).min(axis=2)
predicted_exact = A < np.ldexp(1.0, (quantum_exp + 24).astype(np.int32))
exactness_passed = bool((computed[predicted_exact] == truth[predicted_exact]).all())
inexact_rows = sorted({int(e) for r, e in zip(*np.nonzero(~predicted_exact))})

# Selection stability and score bounds, per row.
softmax_terms = L(C_EXP * U) + 2 * L(U)
selection, score_rows, arithmetic_rows = [], [], []
score_passed = arithmetic_passed = selection_passed = True
for r in range(2):
    S = expected_indices[r]
    m = float(computed[r].max())
    shift = U * np.abs(computed[r] - m)
    rest = [k for k in range(E) if k not in S]
    margin = (min(truth[r, j] - beta[r, j] - shift[j] for j in S)
              - max(truth[r, k] + beta[r, k] + shift[k] for k in rest)
              - 2 * L(C_EXP * U) - 2 * L(U))
    selection.append({'row': r, 'exactGapTenthEleventh': float(sorted(truth[r])[-K] - sorted(truth[r])[-K-1]),
                      'provenMargin': float(margin), 'provenStable': bool(margin > 0)})
    selection_passed &= bool(margin > 0)
    D = max(beta[r, j] + shift[j] for j in S) + L(C_EXP * U) + L(U) + L(gamma(K - 1))
    D0 = max(shift[j] for j in S) + L(C_EXP * U) + L(U) + L(gamma(K - 1))
    retained = [float(computed[r, j]) for j in S]
    exps = [math.exp(v - max(retained)) for v in retained]
    arithmetic_reference = [v / math.fsum(exps) for v in exps]
    diffs, bounds, adiffs, abounds = [], [], [], []
    for c, j in enumerate(S):
        s = expected_scores[r][c]
        bounds.append(s * math.expm1(beta[r, j] + shift[j] + softmax_terms + D) * UP + TRUTH * s)
        diffs.append(abs(float(probabilities[r, c]) - s))
        a = arithmetic_reference[c]
        abounds.append(a * math.expm1(shift[j] + softmax_terms + D0) * UP + TRUTH * a)
        adiffs.append(abs(float(probabilities[r, c]) - a))
    score_passed &= all(d <= b for d, b in zip(diffs, bounds))
    arithmetic_passed &= all(d <= b for d, b in zip(adiffs, abounds))
    w = int(np.argmax(diffs))
    score_rows.append({'row': r, 'maxObserved': max(diffs), 'boundAtMaxObserved': bounds[w],
                       'minBound': min(bounds), 'maxRatio': max(d / b for d, b in zip(diffs, bounds))})
    w = int(np.argmax(adiffs))
    arithmetic_rows.append({'row': r, 'maxObserved': max(adiffs), 'boundAtMaxObserved': abounds[w],
                            'maxRatio': max(d / b for d, b in zip(adiffs, abounds))})

# Diagnostic only: other legitimate F32 summation orders over the same exact products.
P32 = P.astype(np.float32)
sequential_sum = np.cumsum(P32, axis=2, dtype=np.float32)[..., -1].astype(np.float64)
sequential = np.abs(sequential_sum - truth)
pairwise = np.abs(P32.sum(axis=2, dtype=np.float32).astype(np.float64) - truth)
assert (sequential <= logit_bound).all() and (pairwise <= logit_bound).all()

small = {'hidden_size':8,'num_experts':4,'moe_intermediate_size':6,'hidden_act':'silu'}
experts = namespace['Qwen4ExpTextExperts'](SimpleNamespace(**small))
with torch.no_grad():
    for ordinal,parameter in enumerate(experts.parameters()):
        v=((torch.arange(parameter.numel())*(ordinal+3)) % 29 - 14).float()/128
        parameter.copy_(v.reshape(parameter.shape))
    hidden=((torch.arange(24)%11-5).float()/8).reshape(3,8)
    selected_ids=torch.tensor([[0,2],[1,0],[3,4]])
    scores=torch.tensor([[.25,.75],[.625,.375],[1.,0.]])
    output=experts(hidden,selected_ids,scores)
# Independent scalar equations cover weighted scatter, shared expert recipients,
# sparse selections and the upstream sentinel expert slot.
expected=[]
for row in range(3):
    result=[0.]*8
    for slot in range(2):
        expert=int(selected_ids[row,slot])
        if expert==small['num_experts']: continue
        projected=[math.fsum(float(a)*float(b) for a,b in zip(hidden[row],w))
                   for w in experts.gate_up_proj[expert].tolist()]
        middle=[projected[i]/(1+math.exp(-projected[i]))*projected[6+i] for i in range(6)]
        for col,w in enumerate(experts.down_proj[expert].tolist()):
            result[col]+=float(scores[row,slot])*math.fsum(a*b for a,b in zip(middle,w))
    expected.append(result)
expert_error=max(abs(float(output[r,c])-expected[r][c]) for r in range(3) for c in range(8))
expert_passed = expert_error <= 2e-7
report={'schema':'doppler.qwen38-expert-reference/v2','checkpoint':binding['revision'],
        'upstream':binding['referenceImplementation']['revision'],'torch':torch.__version__,
        'environment':{'machine':platform.machine(),'blas':blas.group(1) if blas else None,
                       'cpuCapability':torch.backends.cpu.get_cpu_capability(),
                       'float32MatmulPrecision':torch.get_float32_matmul_precision(),'threads':torch.get_num_threads()},
        'surface':'CPU','scope':'Synthetic router and eager expert body; no learned weights or Doppler execution',
        'routing':{'geometry':{k:config[k] for k in ['hidden_size','num_experts','num_experts_per_tok','norm_topk_prob']},
                   'recipe':'weight[e,h]=(e+1)/1024*((h%13)-6)/128; inputs=base and -base',
                   'indices':record(indices),'scores':record(probabilities),'maxLogitError':logit_error,'maxScoreError':score_error},
        'routerErrorAnalysis':{
            'derivation':ROUTER_DERIVATION,
            'constants':{'u':U,'n':H,'gammaN':gamma(H),'cExp':C_EXP,'k':K,'gammaKMinus1':gamma(K-1)},
            'softmaxSource':{'file':'aten/src/ATen/native/cpu/SoftMaxKernel.cpp','lines':'60-91',
                'url':'https://raw.githubusercontent.com/pytorch/pytorch/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/SoftMaxKernel.cpp',
                'sha256':'8a49d0f097857b201af82c01c0d3da5edbc12fec9eb3bd8c496638069f67dcef',
                'note':'commit is the COMMIT_SHA of the executed torch 2.11.0 build; exp is Vectorized<float>::exp, Sleef_expf4_u10 under AT_BUILD_ARM_VEC256_WITH_SLEEF'},
            'logits':{'maxObserved':float(logit_diff.max()),'at':{'row':int(worst[0]),'expert':int(worst[1])},
                      'boundAtMaxObserved':float(logit_bound[worst]),'absSumAtMaxObserved':float(A[worst]),
                      'maxBound':float(logit_bound.max()),'maxRatio':float(logit_ratio.max())},
            'exactness':{'predictedExactLogits':int(predicted_exact.sum()),'mayRoundLogits':int((~predicted_exact).sum()),
                         'mayRoundExperts':inexact_rows,'nonzeroErrorLogits':int((logit_diff > 0).sum()),
                         'nonzeroErrorsAllInMayRoundSet':bool(((logit_diff > 0) <= ~predicted_exact).all())},
            'selection':selection,'scores':score_rows,
            'scoreArithmeticOnly':{'note':'scores against exact renormalized softmax of the computed F32 logits; isolates softmax/topk/renormalization rounding',
                                   'rows':arithmetic_rows},
            'orderDiagnostic':{'note':'numpy F32 sums of the same exact products; legitimate orders, not upstream execution',
                               'sequentialMax':float(sequential.max()),'pairwiseMax':float(pairwise.max()),
                               'executedEqualsSequentialBitwise':bool((sequential_sum == computed).all())},
            'conclusion':'Observed F32 router error is within the derived bound; the initial 2e-6/2e-7 router tolerances were not derived and are superseded by the elementwise bounds above.'},
        'experts':{'geometry':small,'implementation':'unchanged upstream eager class body; accelerated decorator bypassed explicitly',
                   'weightRecipe':'((arange*(ordinal+3))%29-14)/128',
                   'input':record(hidden),'indices':record(selected_ids),'scores':record(scores),
                   'output':record(output),'independentExpected':expected,'maxError':expert_error},
        'pending':['Full checkpoint expert geometry and learned weights','Shared-expert gate composition',
                   'Accelerated expert implementations','Doppler operators and distributed routing qualification'],
        'checks':{'routerLogits':{'bound':'elementwise gamma_2560*sum|x_i w_ei| plus truth allowance','passed':logit_passed},
                  'routerLogitExactness':{'bound':'bitwise equality where exactness is proven','passed':exactness_passed},
                  'routerSelection':{'bound':'proven top-k stability margin > 0','passed':selection_passed},
                  'routerScores':{'bound':'elementwise s_i*expm1(N_i+D) plus truth allowance','passed':score_passed},
                  'routerScoreArithmetic':{'bound':'same with beta = 0 against softmax of computed logits','passed':arithmetic_passed},
                  'expertOutput':{'tolerance':2e-7,'basis':'initial fixture tolerance, not derived','passed':expert_passed}},
        'supersededRouterTolerances':{'logits':2e-6,'scores':2e-7,'reason':'chosen without derivation; below the error that legitimate F32 summation orders produce at these magnitudes'},
        'modelAcceptanceThresholdsChanged':False}
report['passed'] = all(check['passed'] for check in report['checks'].values())
report['modelQualified'] = False
(ROOT/'expert-reference-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'logitError':logit_error,'logitBound':float(logit_bound[worst]),
                  'scoreError':score_error,'expertError':expert_error,'modelQualified':False}))

if not report['passed']: raise SystemExit(1)
