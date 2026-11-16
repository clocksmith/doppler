"""Source-pinned client DeltaNet graph with request-local cache and projection boundary.

This is a client reference/operator adapter, not qualified Flash weights, encrypted
execution or production cryptographic support. A callback is trusted client code;
it must independently bind and verify any remote encrypted operations it performs.
"""
import torch
from torch import nn
from flash_reference import load_reference
from flash_delta_state import DeltaRequestState
from flash_delta_program import ORDER
from flash_client_projections import ClientProjection,client_execution

class ClientDeltaNet(nn.Module):
    def __init__(self,reference_path,config,layer_index=0):
        super().__init__()
        source=load_reference(reference_path,'delta')
        self.model=source['Qwen4ExpTextGatedDeltaNet'](config,layer_index).eval().requires_grad_(False)
        self.layer_index=layer_index;self.hidden_width=config.hidden_size
        for name in ORDER:
            layer=getattr(self.model,name)
            setattr(self.model,name,ClientProjection(name,layer))

    def forward(self,value,state,context,*,execution,verified_client_projection=None,cancelled=None,attention_mask=None):
        if not isinstance(state,DeltaRequestState):raise ValueError('Request-owned DeltaNet state required')
        with client_execution(state,context,ORDER,execution,verified_client_projection,cancelled):
            state.require(state.layer_index==self.layer_index and value.dtype==torch.float32 and value.ndim==3 and value.shape[0]==1
                          and value.shape[-1]==self.hidden_width and value.shape[1]>0 and torch.isfinite(value).all())
            with torch.no_grad():result=self.model(value.clone(),cache=state,attention_mask=attention_mask)
            state.complete(value.shape[1]);return result
