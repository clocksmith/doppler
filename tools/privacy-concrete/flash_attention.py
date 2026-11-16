"""Inspected eager QSA attention client graph; no actual-weight or encrypted qualification."""
import json
from pathlib import Path
import torch
from torch import nn
from flash_reference import load_reference
from flash_client_projections import ClientProjection,client_execution
from flash_attention_state import AttentionRequestState

CONFIG=json.loads(Path(__file__).with_name('flash-attention-program.json').read_text())
ORDER=tuple(CONFIG['projectionOrder'])


class ClientQSAAttention(nn.Module):
    def __init__(self,reference_path,config,*,context_limit,rotary_width,layer_index=0):
        super().__init__()
        if (config._attn_implementation!=CONFIG['attentionImplementation'] or type(context_limit) is not int
                or not 0<context_limit<=config.max_position_embeddings or type(rotary_width) is not int
                or rotary_width<=0 or rotary_width%2 or rotary_width>min(config.head_dim,config.indexer_head_dim)):
            raise ValueError('Explicit bounded inspected eager-attention configuration required')
        source=load_reference(reference_path,'attention')
        self.model=source['Qwen4ExpTextAttention'](config,layer_index).eval().requires_grad_(False)
        self.hidden_width=config.hidden_size;self.context_limit=context_limit;self.rotary_width=rotary_width;self.layer_index=layer_index
        for name in ORDER:
            parent,_,leaf=name.rpartition('.')
            owner=self.model.get_submodule(parent) if parent else self.model
            setattr(owner,leaf,ClientProjection(name,getattr(owner,leaf)))

    def forward(self,value,state,context,*,position_embeddings,attention_mask,execution,verified_client_projection=None,cancelled=None):
        if not isinstance(state,AttentionRequestState):raise ValueError('Request-owned full-attention state required')
        with client_execution(state,context,ORDER,execution,verified_client_projection,cancelled):
            state.require(state.layer_index==self.layer_index and value.dtype==torch.float32 and value.ndim==3 and value.shape[0]==1
                          and value.shape[-1]==self.hidden_width and value.shape[1]>0 and torch.isfinite(value).all())
            total=state.tokens+value.shape[1]
            state.require(total<=self.context_limit and attention_mask.dtype==torch.float32 and tuple(attention_mask.shape)==(1,1,value.shape[1],total)
                          and len(position_embeddings)==2
                          and all(tuple(item.shape)==(1,total,self.rotary_width) and item.dtype==torch.float32 and torch.isfinite(item).all() for item in position_embeddings))
            visible=attention_mask[0,0]==0
            causal=torch.arange(total)[None,:]<=state.tokens+torch.arange(value.shape[1])[:,None]
            state.require(torch.all((attention_mask==0)|(attention_mask==torch.finfo(torch.float32).min))
                          and visible.any(-1).all() and not (visible&~causal).any())
            with torch.no_grad():result,_=self.model(value.clone(),position_embeddings=position_embeddings,attention_mask=attention_mask.clone(),cache=state)
            state.complete(value.shape[1]);return result
