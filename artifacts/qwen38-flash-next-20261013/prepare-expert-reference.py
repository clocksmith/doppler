"""Pinned upstream eager expert controls; synthetic CPU fixtures, not compatibility."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
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

router = namespace['Qwen4ExpTextTopKRouter'](SimpleNamespace(**config))
base = (torch.arange(config['hidden_size']) % 13 - 6).float() / 128
scale = (torch.arange(config['num_experts']) + 1).float() / 1024
with torch.no_grad():
    router.weight.copy_(scale[:,None] * base[None,:])
    inputs = torch.stack([base, -base])
    logits, probabilities, indices = router(inputs)
expected_indices = [list(range(config['num_experts']-1, config['num_experts']-config['num_experts_per_tok']-1, -1)),
                    list(range(config['num_experts_per_tok']))]
assert indices.tolist() == expected_indices
expected_logits = [[math.fsum(float(a)*float(b) for a,b in zip(row,weight))
                    for weight in router.weight.tolist()] for row in inputs.tolist()]
logit_error = max(abs(float(logits[r,c])-expected_logits[r][c]) for r in range(2) for c in range(config['num_experts']))
logit_passed = logit_error <= 2e-6
expected_scores=[]
for r in range(2):
    retained=[expected_logits[r][i] for i in expected_indices[r]]
    exps=[math.exp(v-max(retained)) for v in retained]
    expected_scores.append([v/math.fsum(exps) for v in exps])
score_error=max(abs(float(probabilities[r,c])-expected_scores[r][c]) for r in range(2) for c in range(config['num_experts_per_tok']))
score_passed = score_error <= 2e-7

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
report={'schema':'doppler.qwen38-expert-reference/v1','checkpoint':binding['revision'],
        'upstream':binding['referenceImplementation']['revision'],'torch':torch.__version__,
        'surface':'CPU','scope':'Synthetic router and eager expert body; no learned weights or Doppler execution',
        'routing':{'geometry':{k:config[k] for k in ['hidden_size','num_experts','num_experts_per_tok','norm_topk_prob']},
                   'recipe':'weight[e,h]=(e+1)/1024*((h%13)-6)/128; inputs=base and -base',
                   'indices':record(indices),'scores':record(probabilities),'maxLogitError':logit_error,'maxScoreError':score_error},
        'experts':{'geometry':small,'implementation':'unchanged upstream eager class body; accelerated decorator bypassed explicitly',
                   'weightRecipe':'((arange*(ordinal+3))%29-14)/128',
                   'input':record(hidden),'indices':record(selected_ids),'scores':record(scores),
                   'output':record(output),'independentExpected':expected,'maxError':expert_error},
        'pending':['Full checkpoint expert geometry and learned weights','Shared-expert gate composition',
                   'Accelerated expert implementations','Doppler operators and distributed routing qualification'],
        'checks':{'routerLogits':{'tolerance':2e-6,'passed':logit_passed},'routerScores':{'tolerance':2e-7,'passed':score_passed},'expertOutput':{'tolerance':2e-7,'passed':expert_passed}},
        'passed':logit_passed and score_passed and expert_passed,'modelQualified':False}
(ROOT/'expert-reference-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'routerError':score_error,'expertError':expert_error,'modelQualified':False}))

if not report['passed']: raise SystemExit(1)
