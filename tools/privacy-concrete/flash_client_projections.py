"""Per-request projection execution shared by inspected Flash client graphs."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
import torch
from torch import nn
from flash_request_state import ClientRequestState

_CURRENT=ContextVar('doppler_flash_client_request',default=None)


@dataclass(frozen=True)
class ProjectionExecution:
    state:ClientRequestState
    callback:object


def current_client_state():
    execution=_CURRENT.get()
    if execution is None:raise ValueError('No bound Flash client execution')
    execution.state.require(execution.state.active)
    return execution.state


class ClientProjection(nn.Module):
    def __init__(self,name,source):
        super().__init__();self.source=source
        self.operation=MappingProxyType(dict(id=name,inputWidth=source.in_features,outputWidth=source.out_features))

    def forward(self,value):
        execution=_CURRENT.get()
        if execution is None:raise ValueError('No bound Flash client execution')
        state=execution.state;state.projection(self.operation['id'])
        state.require(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.ndim>=2
                      and value.shape[-1]==self.operation['inputWidth'] and torch.isfinite(value).all())
        owned=value.clone()
        result=self.source(owned) if execution.callback is None else execution.callback(self.operation,owned)
        state.require(tuple(result.shape)==(*value.shape[:-1],self.operation['outputWidth']) and torch.isfinite(result).all())
        owned_result=result.detach().clone()
        state.verified()
        return owned_result



@contextmanager
def client_execution(state,context,order,execution,callback,cancelled):
    if not isinstance(state,ClientRequestState):raise ValueError('Request-owned Flash state required')
    try:
        if execution=='public-reference':
            if callback is not None:raise ValueError('Reference mode does not accept a projection adapter')
        elif execution=='verified-projections':
            if not callable(callback):raise ValueError('Explicit trusted client projection adapter required')
        else:raise ValueError('Unsupported Flash client execution mode')
    except BaseException:
        state.fail();raise
    state.begin(context,order,cancelled,execution);token=_CURRENT.set(ProjectionExecution(state,callback))
    try:yield
    except BaseException:
        state.fail();raise
    finally:_CURRENT.reset(token)
