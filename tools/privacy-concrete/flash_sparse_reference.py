"""Actual-weight official sparse MLP reference, using only verified local public expert slices.

This is a trusted-client clear reference, never a selected-only provider path.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from flash_reference import load_reference
from flash_moe_plan import PROGRAM as MOE_PROGRAM

MOE_PREFIX='model.language_model.layers.0.mlp.'
HYPER_PREFIX='model.language_model.layers.0.mlp_hyper_connection.'


class LocalExpertBank:
    def __init__(self,source,name):self.source=source;self.name=name
    def __getitem__(self,index):return self.source.tensor(self.name,expert=int(index))


def source_mlp_reference(moe_source,hyper_source,reference_path):
    declarations=load_reference(reference_path,'moe')
    text=dict(moe_source.config['text_config'])
    resolved=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())['resolvedOperatorDefaults']
    if 'norm_topk_prob' not in text:text['norm_topk_prob']=resolved['normTopKProbability']
    if type(text['norm_topk_prob']) is not bool or text['norm_topk_prob']!=MOE_PROGRAM.normalize_topk:
        raise ValueError('Actual source routing normalization differs from the admitted client program')
    config=SimpleNamespace(**text)
    with torch.device('meta'):
        moe=declarations['Qwen4ExpTextSparseMoeBlock'](config)
        hyper=declarations['Qwen4ExpTextGatedResidual'](config)
    for name in ['gate.weight','shared_expert_gate.weight','shared_expert.gate_proj.weight','shared_expert.up_proj.weight','shared_expert.down_proj.weight']:
        parent,_,field=name.rpartition('.')
        setattr(moe.get_submodule(parent),field,nn.Parameter(moe_source.tensor(MOE_PREFIX+name),requires_grad=False))
    for name in ['gate_up_proj','down_proj']:
        del moe.experts._parameters[name]
        setattr(moe.experts,name,LocalExpertBank(moe_source,MOE_PREFIX+'experts.'+name))
    for name in ['hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight','block_inject_weight.weight']:
        parent,_,field=name.rpartition('.')
        setattr(hyper.get_submodule(parent),field,nn.Parameter(hyper_source.tensor(HYPER_PREFIX+name),requires_grad=False))
    moe.eval();hyper.eval()
    def evaluate(value):
        shaped=value.reshape(1,1,-1)
        with torch.no_grad():
            mixed,residual,injection=hyper(shaped)
            output=residual+(moe(mixed).unsqueeze(-2)*injection.unsqueeze(-1)).flatten(-2)
        return output.reshape_as(value)
    return evaluate
