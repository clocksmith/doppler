"""Tensor-free identities for evictable, trusted-client public preparation.

This is not a serialized proof/import interface. The record is derived from a
real owned FullLinear, and every rematerialization must match before use.
"""
from dataclasses import dataclass
import hashlib
import json
from full_plan import canonical


def public_metadata_digest(projection):
    import torch
    result=hashlib.sha256(canonical(dict(projection=projection.definition,encoding=projection.encoding)))
    for name in ('scale','zero_point','sum_weights','prepared_bias'):
        value=getattr(projection,name)
        if value is None:
            if name!='prepared_bias':raise ValueError('Complete public quantizer metadata required')
            result.update(canonical(dict(name=name,value=None)));continue
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.numel()!=projection.definition['outputWidth'] or not torch.isfinite(value).all()):
            raise ValueError('Finite CPU per-output quantizer metadata required')
        result.update(canonical(dict(name=name,shape=list(value.shape),dtype=str(value.dtype))))
        with memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()) as raw:result.update(raw)
    return result.hexdigest()


@dataclass(frozen=True)
class ProjectionIdentity:
    encoded_definition:bytes
    encoded_encoding:bytes
    metadata_digest:str

    @classmethod
    def from_projection(cls,projection):
        from full_model import FullLinear
        if type(projection) is not FullLinear:raise ValueError('Owned public linear preparation required')
        return cls(canonical(projection.definition),canonical(projection.encoding),public_metadata_digest(projection))

    @property
    def definition(self):return json.loads(self.encoded_definition)

    @property
    def encoding(self):return json.loads(self.encoded_encoding)

    def require_projection(self,projection):
        if (canonical(projection.definition)!=self.encoded_definition or canonical(projection.encoding)!=self.encoded_encoding
                or public_metadata_digest(projection)!=self.metadata_digest):
            raise ValueError('Rematerialized public projection differs from trusted client preparation')
