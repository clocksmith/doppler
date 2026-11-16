"""Trusted-client Flash grouped normalization, hyperconnection and residual composition."""
import torch
from torch import nn
from flash_hyper_program import HyperProgram




class TorchHyperConnection(nn.Module):
    """Projection callback is a trusted local adapter returning verified local plaintext.

    This graph is not a network service, encryption implementation or security qualifier.
    RMSNorm uses the source's (1 + weight) convention and per-stream grouping.
    """
    def __init__(self,program,norm_weight,verified_client_projection):
        super().__init__();self.program=program;self.project=verified_client_projection
        width=program.streams*program.hidden_width
        if tuple(norm_weight.shape)!=(width,) or not torch.isfinite(norm_weight).all():
            raise ValueError('Explicit finite hyperconnection normalization weights required')
        self.register_buffer('norm_weight',norm_weight.detach().clone())

    def forward(self,value):
        program=self.program;width=program.streams*program.hidden_width
        if value.ndim<2 or value.shape[-1]!=width or not torch.isfinite(value).all():
            raise ValueError('Explicit finite hyperconnection input geometry required')
        owned=value.clone()
        grouped=owned.float().unflatten(-1,(program.streams,program.hidden_width))
        normalized=grouped*torch.rsqrt(grouped.square().mean(-1,keepdim=True)+program.epsilon)
        normalized=(normalized.flatten(-2)*(1.0+self.norm_weight.float())).to(owned.dtype)
        operations=program.contract()['operations']
        down=self.project(operations[0],normalized.clone())
        if tuple(down.shape)!=(*owned.shape[:-1],program.low_rank):raise ValueError('Hyperconnection down geometry changed')
        up=self.project(operations[1],torch.nn.functional.silu(down/program.streams))
        if tuple(up.shape)!=tuple(owned.shape):raise ValueError('Hyperconnection up geometry changed')
        mixed=(torch.sigmoid(up).unflatten(-1,(program.streams,program.hidden_width))
               *normalized.unflatten(-1,(program.streams,program.hidden_width))).mean(-2)
        inject=self.project(operations[2],normalized.clone())
        if tuple(inject.shape)!=(*owned.shape[:-1],program.streams):raise ValueError('Hyperconnection injection geometry changed')
        injection=2*torch.sigmoid(inject/program.streams)
        if not torch.isfinite(mixed).all() or not torch.isfinite(injection).all():raise ValueError('Finite verified hyperconnection output required')
        return mixed,owned,injection

    def combine(self,residual,injection,block_output):
        program=self.program
        if (tuple(residual.shape[-1:])!=(program.streams*program.hidden_width,)
                or tuple(injection.shape)!=(*residual.shape[:-1],program.streams)
                or tuple(block_output.shape)!=(*residual.shape[:-1],program.hidden_width)
                or not torch.isfinite(block_output).all()):
            raise ValueError('Explicit hyperconnection residual geometry required')
        result=residual+(block_output.unsqueeze(-2)*injection.unsqueeze(-1)).flatten(-2)
        if not torch.isfinite(result).all():raise ValueError('Finite hyperconnection residual required')
        return result

    def mlp_branch(self,value,verified_client_moe):
        mixed,residual,injection=self(value)
        return self.combine(residual,injection,verified_client_moe(mixed))
