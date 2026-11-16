"""Request-bound Torch adapter over verified, shared, entirely local BF16 table files."""
import torch
from torch import nn
from local_ngram_rows import LocalNgramRows
from flash_client_projections import current_client_state
from flash_ple_state import PLERequestState


class LocalNgramEmbedding(nn.Module):
    def __init__(self,storage,*,max_lookup_bytes,layer_index):
        super().__init__()
        if type(storage) is not LocalNgramRows or type(max_lookup_bytes) is not int or max_lookup_bytes<=0:
            raise ValueError('Verified local row storage and explicit lookup budget required')
        storage._unchanged()
        self.storage=storage;self.max_lookup_bytes=max_lookup_bytes;self.layer_index=layer_index
        self.num_embeddings=storage.rows;self.embedding_dim=storage.columns
        # The pinned source queries only weight.device. No fake full table is allocated.
        self.register_buffer('weight',torch.empty(0,dtype=torch.float32))

    def forward(self,indices):
        state=current_client_state().ple_state_for(self.layer_index)
        state.require(isinstance(state,PLERequestState) and indices.dtype==torch.int64
                      and indices.device.type=='cpu' and indices.numel()>0)
        if state.local_rows_lease is None:state.local_rows_lease=self.storage.lease(state.context)
        lease=state.local_rows_lease
        state.require(lease.storage is self.storage)
        data=lease.read_rows(indices.reshape(-1).tolist(),state.context,max_output_bytes=self.max_lookup_bytes,cancelled=state.cancelled)
        result=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(*indices.shape,self.embedding_dim).to(dtype=torch.float32)
        state.require(torch.isfinite(result).all())
        return result
