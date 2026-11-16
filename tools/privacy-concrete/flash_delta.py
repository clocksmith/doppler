"""Source-pinned client DeltaNet graph with request-local cache and projection boundary.

This is a client reference/operator adapter, not qualified Flash weights, encrypted
execution or production cryptographic support. A callback is trusted client code;
it must independently bind and verify any remote encrypted operations it performs.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
import torch
from torch import nn
from flash_reference import load_reference
from flash_delta_state import DeltaRequestState

_CURRENT=ContextVar('doppler_flash_delta_request',default=None)
ORDER=('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')


@dataclass(frozen=True)
class ProjectionExecution:
    state:DeltaRequestState
    callback:object


class ClientDeltaProjection(nn.Module):
    def __init__(self,name,source):
        super().__init__();self.source=source
        self.operation=MappingProxyType(dict(id=name,inputWidth=source.in_features,outputWidth=source.out_features))

    def forward(self,value):
        execution=_CURRENT.get()
        if execution is None:raise ValueError('No bound DeltaNet client execution')
        state=execution.state;state.projection(self.operation['id'])
        owned=value.clone()
        result=self.source(owned) if execution.callback is None else execution.callback(self.operation,owned)
        state.require(tuple(result.shape)==(*value.shape[:-1],self.operation['outputWidth']) and torch.isfinite(result).all())
        owned_result=result.detach().clone()
        state.verified()
        return owned_result


class ClientDeltaNet(nn.Module):
    def __init__(self,reference_path,config,layer_index=0):
        super().__init__()
        source=load_reference(reference_path,'delta')
        self.model=source['Qwen4ExpTextGatedDeltaNet'](config,layer_index).eval().requires_grad_(False)
        self.layer_index=layer_index;self.hidden_width=config.hidden_size
        for name in ORDER:
            layer=getattr(self.model,name)
            setattr(self.model,name,ClientDeltaProjection(name,layer))

    def forward(self,value,state,context,*,execution,verified_client_projection=None,cancelled=None,attention_mask=None):
        # Explicit reference execution cannot silently stand in for a missing encrypted adapter.
        if not isinstance(state,DeltaRequestState):raise ValueError('Request-owned DeltaNet state required')
        try:
            if execution=='public-reference':
                if verified_client_projection is not None:raise ValueError('Reference mode does not accept an external projection adapter')
            elif execution=='verified-projections':
                if not callable(verified_client_projection):raise ValueError('An explicit trusted client projection adapter is required')
            else:raise ValueError('Unsupported client DeltaNet execution mode')
        except BaseException:
            state.fail();raise
        state.begin(context,ORDER,cancelled,execution);token=_CURRENT.set(ProjectionExecution(state,verified_client_projection))
        try:
            state.require(state.layer_index==self.layer_index and value.ndim==3 and value.shape[0]==1
                          and value.shape[-1]==self.hidden_width and value.shape[1]>0 and torch.isfinite(value).all())
            with torch.no_grad():result=self.model(value.clone(),cache=state,attention_mask=attention_mask)
            state.complete(value.shape[1]);return result
        except BaseException:
            state.fail();raise
        finally:_CURRENT.reset(token)
