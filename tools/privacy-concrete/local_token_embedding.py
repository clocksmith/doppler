"""Request-owned Torch input embedding without full-table FP32 materialization."""
import torch
from torch import nn
from local_token_rows import LocalTokenRows
from flash_client_projections import current_client_state
from flash_model_state import FlashModelState


class LocalTokenEmbedding(nn.Module):
    def __init__(self,storage,*,max_lookup_rows):
        super().__init__()
        if type(storage) is not LocalTokenRows or type(max_lookup_rows) is not int or max_lookup_rows<=0:
            raise ValueError('Verified token embedding storage and explicit lookup rows required')
        storage._unchanged();self.storage=storage;self.max_lookup_rows=max_lookup_rows
        self.num_embeddings=storage.rows;self.embedding_dim=storage.columns

    def forward(self,indices):
        state=current_client_state()
        state.require(type(state) is FlashModelState and indices.dtype==torch.int64 and indices.device.type=='cpu'
                      and indices.ndim==2 and indices.shape[0]==1 and 0<indices.numel()<=self.max_lookup_rows)
        if state.token_embedding_lease is None:state.token_embedding_lease=self.storage.lease(state.context)
        lease=state.token_embedding_lease;state.require(lease.storage is self.storage and lease.context==state.context)
        data=lease.read_rows(indices.reshape(-1).tolist(),state.context,max_output_bytes=self.max_lookup_rows*self.embedding_dim*2,cancelled=state.cancelled)
        result=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(*indices.shape,self.embedding_dim).float()
        state.require(torch.isfinite(result).all());return result
