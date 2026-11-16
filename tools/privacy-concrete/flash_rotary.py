"""Inspected pinned default text-only rotary methods; no dynamic/multimodal fallback."""
import ast
import __future__
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from flash_reference import SOURCE


def text_rotary(reference_path,config,total):
    if (type(total) is not int or not 0<total<=config.max_position_embeddings or config.rope_parameters['rope_type']!='default'):
        raise ValueError('Bounded inspected default text rotary required')
    with Path(reference_path).open('rb') as handle:raw=handle.read(1024*1024+1)
    if hashlib.sha256(raw).hexdigest()!=SOURCE['files']['modeling_qwen4_exp.py']['sha256']:raise ValueError('Pinned rotary reference required')
    root=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpTextRotaryEmbedding')
    selected={n.name:n for n in root.body if isinstance(n,ast.FunctionDef) and n.name in ['compute_default_rope_parameters','recomposition_frequencies']}
    if len(selected)!=2:raise ValueError('Inspected default rotary methods changed')
    for node in selected.values():node.decorator_list=[]
    scope=dict(torch=torch)
    exec(compile(ast.Module(body=list(selected.values()),type_ignores=[]),'pinned-default-text-rotary','exec',flags=__future__.annotations.compiler_flag),scope)
    inv_freq,scaling=scope['compute_default_rope_parameters'](config)
    positions=torch.arange(total,dtype=torch.int64).reshape(1,1,total).expand(3,1,total)
    freqs=positions[...,None].float()*inv_freq.to(device='cpu',dtype=torch.float)
    owner=SimpleNamespace(mrope_section=config.rope_parameters['mrope_section'])
    return tuple(scope['recomposition_frequencies'](owner,freq)*scaling for freq in [freqs.cos(),freqs.sin()])
