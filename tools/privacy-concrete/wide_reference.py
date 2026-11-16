"""Clear-only high-precision control. No encryption/backend registration or narrow truncation."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from full_model import prepare
from full_plan import canonical
from protocol import digest
from integer_reference import integer_dot
from public_weights import mapped_public_matrix,store_public_matrix


class WideReferenceLinear(nn.Module):
    def __init__(self,original,executor,directory,bits):
        super().__init__();self.name=original.name;self.source=original.remote;self.executor=executor;self.mode='float-reference'
        if bits!=12:raise ValueError('Only the explicit clear-only 12-bit control is installed')
        self.context=SimpleNamespace(input_quantizers=[SimpleNamespace(quant_options=SimpleNamespace(n_bits=bits))])
        q,self.scale,self.zero,self.sum_weights=executor._per_channel_weight_quantization(self.source.weight.detach().numpy().T,self.context,torch.device('cpu'))
        values=np.ascontiguousarray(q.numpy(),dtype=np.int16)
        if not np.array_equal(q.numpy(),values):raise ValueError('Wide reference coefficient truncation')
        self.weight_path,definition=store_public_matrix(directory,self.name,self.source.in_features,self.source.out_features,
            dict(weightEncoding='signed-int12-in-int16',quantizationBits=bits),values.tobytes())
        self.definition=dict(definition,quantizationBits=bits)

    def coefficient_matrix(self):
        return mapped_public_matrix(self.weight_path,(self.definition['inputWidth'],self.definition['outputWidth']),'<i2')

    def forward(self,value):
        if self.mode=='float-reference':return self.source(value)
        if self.mode!='clear-reference':raise ValueError('Clear-only reference cannot encrypt or silently substitute a backend')
        q,scale,zero,shape=self.executor._dynamic_input_quantization(value,self.context,False)
        with self.coefficient_matrix() as weights:raw=integer_dot(q,weights)
        result=self.executor._apply_correction_and_dequantize(raw,q,zero,self.zero,self.sum_weights,self.definition['inputWidth'],scale,self.scale)
        result=result.view(*shape[:-1],self.definition['outputWidth'])
        return result if self.source.bias is None else result+self.source.bias


def prepare_wide(model_dir,cache_dir):
    model,tokenizer,executor,old,base=prepare(model_dir,cache_dir)
    projections=[]
    for original in old:
        replacement=WideReferenceLinear(original,executor,cache_dir,12)
        owner,_,leaf=original.name.rpartition('.')
        setattr(model.get_submodule(owner) if owner else model,leaf,replacement);projections.append(replacement)
    identity=digest(canonical(dict(source=base.definition()['source'],arithmetic='clear-int64-quantization12/v1',projections=[p.definition for p in projections])))
    return model,tokenizer,projections,identity
