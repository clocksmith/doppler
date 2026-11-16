"""Bind an inspected complete decoder to an owned, authenticated public stage.

Linear matrices stream from source; nonlinear parameters stay client-local.
This adds no encrypted plan, provider admission, or full-model qualification.
"""
from pathlib import Path
import torch
from torch import nn
from flash_reference import load_reference
from flash_text_config import resolve_text_config
from flash_source_scopes import staging_spec
from flash_streamed_moe import ProjectionSource,SourceLinear,SourceDenseClientMoE
from flash_weights import FlashSource
from local_ngram_embedding import LocalNgramEmbedding
from flash_ple_plan import TABLE
from full_plan import canonical
from protocol import require_active,digest,parse_unique_json


def bind_decoder(source,reference_path,layer_index,*,local_rows=None,max_lookup_bytes=None,cancelled=None):
    require_active(cancelled)
    if not isinstance(source,FlashSource) or canonical(source.spec)!=canonical(staging_spec('decoder',layer_index)):
        raise ValueError('Exact verified decoder-stage source required')
    config=source_config(source,reference_path)
    declarations=load_reference(reference_path,'decoder')
    with torch.device('meta'):layer=declarations['Qwen4ExpTextDecoderLayer'](config,layer_index)
    owner=ProjectionSource(source);prefix=f'model.language_model.layers.{layer_index}.'
    consumed=set()
    if layer.ple is not None:
        if local_rows is None or canonical(local_rows.spec)!=canonical(TABLE):
            raise ValueError('Complete installed local PLE row store required')
        layer.ple.ple_embedding.ngram_embedding=LocalNgramEmbedding(local_rows,max_lookup_bytes=max_lookup_bytes,layer_index=layer_index)
    elif local_rows is not None:raise ValueError('Non-PLE decoder does not accept a row store')
    layer.mlp=SourceDenseClientMoE(source,config,layer_index)
    consumed.update(module._name for module in layer.mlp.ops.values())
    for name,module in list(layer.named_modules()):
        require_active(cancelled)
        if name.startswith('mlp.') or not isinstance(module,nn.Linear):continue
        if module.bias is not None:raise ValueError('Unexpected biased Flash projection')
        tensor=prefix+name+'.weight'
        replacement=SourceLinear(source,tensor,module.in_features,module.out_features,source_owner=owner)
        parent,_,leaf=name.rpartition('.');setattr(layer.get_submodule(parent) if parent else layer,leaf,replacement)
        consumed.add(tensor)
    for name,value in list(layer.named_parameters())+list(layer.named_buffers()):
        require_active(cancelled)
        if name.startswith('mlp.') or (layer.ple is not None and name.startswith('ple.ple_embedding.ngram_embedding.')):continue
        tensor=prefix+name
        if tensor in consumed:continue
        if tensor not in source.spec['tensors']:raise ValueError('Unaccounted decoder client parameter')
        loaded=source.tensor(tensor)
        if loaded.shape!=value.shape or loaded.dtype!=value.dtype:raise ValueError('Decoder client parameter geometry/dtype changed')
        parent,_,leaf=name.rpartition('.');module=layer.get_submodule(parent) if parent else layer
        setattr(module,leaf,nn.Parameter(loaded,requires_grad=False) if leaf in module._parameters else loaded)
        consumed.add(tensor)
    if consumed!=set(source.spec['tensors']):raise ValueError('Incomplete decoder stage inventory consumption')
    owner.require_source();require_active(cancelled)
    layer.eval().requires_grad_(False)
    return layer,config


def source_config(source,reference_path):
    raw=(Path(source.directory)/'config.json').read_bytes();record=source.spec['files']['config.json']
    if len(raw)!=record['bytes'] or digest(raw)!=record['sha256']:raise ValueError('Pinned decoder configuration required')
    return resolve_text_config(parse_unique_json(raw)['text_config'],
        Path(reference_path).with_name('configuration_qwen4_exp.py').read_bytes(),attention_implementation='eager')
