"""Compose a full Flash client graph from caller-owned authenticated stages.

No downloads or ownership transfer. Every stage and both local row stores must
remain live; this does not implement staged eviction or provision coefficient
storage. Linear matrices remain source-bound rather than retained in FP32.
"""
import torch
from torch import nn
from flash_decoder_source import bind_decoder,source_config
from flash_source_scopes import staging_spec,TEXT
from flash_reference import load_reference
from flash_streamed_moe import SourceLinear,ProjectionSource
from flash_weights import FlashSource
from flash_model_graph import ClientFlashGraph
from local_token_embedding import LocalTokenEmbedding
from local_token_rows import installed_inventory
from flash_ple_plan import TABLE
from full_plan import canonical
from protocol import require_active


def bind_output(source,reference_path,*,cancelled=None):
    require_active(cancelled)
    if not isinstance(source,FlashSource) or canonical(source.spec)!=canonical(staging_spec('output')):
        raise ValueError('Exact authenticated Flash output stage required')
    config=source_config(source,reference_path);declarations=load_reference(reference_path,'decoder')
    with torch.device('meta'):
        mixer=declarations['Qwen4ExpTextGatedResidual'](config,use_combine=False)
    owner=ProjectionSource(source);prefix='model.language_model.hyper_connection_mixer.';consumed=set()
    for name,module in list(mixer.named_modules()):
        require_active(cancelled)
        if not isinstance(module,nn.Linear):continue
        if module.bias is not None:raise ValueError('Unexpected biased output projection')
        tensor=prefix+name+'.weight';parent,_,leaf=name.rpartition('.')
        setattr(mixer.get_submodule(parent) if parent else mixer,leaf,
            SourceLinear(source,tensor,module.in_features,module.out_features,source_owner=owner));consumed.add(tensor)
    for name,value in list(mixer.named_parameters())+list(mixer.named_buffers()):
        require_active(cancelled);tensor=prefix+name
        if tensor in consumed:continue
        if tensor not in source.spec['tensors']:raise ValueError('Unaccounted output client parameter')
        loaded=source.tensor(tensor)
        if loaded.shape!=value.shape or loaded.dtype!=value.dtype:raise ValueError('Output client parameter geometry/dtype changed')
        parent,_,leaf=name.rpartition('.');module=mixer.get_submodule(parent) if parent else mixer
        setattr(module,leaf,nn.Parameter(loaded,requires_grad=False) if leaf in module._parameters else loaded);consumed.add(tensor)
    head=SourceLinear(source,'lm_head.weight',config.hidden_size,config.vocab_size,source_owner=owner);consumed.add('lm_head.weight')
    if consumed!=set(source.spec['tensors']):raise ValueError('Incomplete output stage inventory consumption')
    owner.require_source();require_active(cancelled);mixer.eval().requires_grad_(False)
    return mixer,head,config


def bind_graph(decoders,output,token_rows,ple_rows,reference_path,*,context_limit,max_lookup_rows,max_ple_lookup_bytes,cancelled=None):
    require_active(cancelled)
    if type(decoders) not in [list,tuple] or len(decoders)!=TEXT['layerCount']:
        raise ValueError('All installed Flash decoder stages required')
    if canonical(token_rows.spec)!=canonical(installed_inventory()) or canonical(ple_rows.spec)!=canonical(TABLE):
        raise ValueError('Complete installed client-local token and PLE row stores required')
    mixer,head,config=bind_output(output,reference_path,cancelled=cancelled);layers=[]
    for index,source in enumerate(decoders):
        layer,layer_config=bind_decoder(source,reference_path,index,
            local_rows=ple_rows if index+1 in config.ple_layer_ids else None,max_lookup_bytes=max_ple_lookup_bytes,cancelled=cancelled)
        if canonical(vars(layer_config))!=canonical(vars(config)):raise ValueError('Decoder and output configuration identities differ')
        layers.append(layer)
    embedding=LocalTokenEmbedding(token_rows,max_lookup_rows=max_lookup_rows)
    result=ClientFlashGraph(reference_path,config,embedding,layers,mixer,head,context_limit=context_limit)
    require_active(cancelled)
    return result
