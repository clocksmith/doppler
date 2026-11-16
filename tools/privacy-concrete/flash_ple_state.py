"""Client-owned PLE token history and dilated-convolution state."""
import torch
from torch.nn import functional as F
from flash_request_state import ClientRequestState


class PLERequestState(ClientRequestState):
    def __init__(self,context,layer_index=1):
        super().__init__(context,layer_index);self.conv={};self.local_rows_lease=None

    def clear_private_state(self):
        self.conv.clear()
        if self.local_rows_lease is not None:self.local_rows_lease.close()
        self.local_rows_lease=None

    def ple_state_for(self,layer_idx):
        self.require(self.active and layer_idx==self.layer_index);return self

    def _slot(self,layer_idx,state_idx=0):
        self.require(self.active and type(layer_idx) is int and layer_idx==self.layer_index
                     and type(state_idx) is int and state_idx in [1,2])
        return layer_idx,state_idx

    def get_conv_state(self,layer_idx,state_idx=0):return self.conv.get(self._slot(layer_idx,state_idx))

    def update_conv_state(self,value,layer_idx,state_idx=0,conv_kernel_size=None):
        slot=self._slot(layer_idx,state_idx)
        self.require(type(conv_kernel_size) is int and conv_kernel_size>0
                     and ((state_idx==1 and value.ndim==3) or (state_idx==2 and value.ndim==2 and value.dtype==torch.int64)))
        previous=self.conv.get(slot)
        joined=value if previous is None else torch.cat([previous,value],dim=-1)
        owned=joined[...,-conv_kernel_size:].detach().clone()
        if owned.shape[-1]<conv_kernel_size:owned=F.pad(owned,(conv_kernel_size-owned.shape[-1],0))
        self.conv[slot]=owned
        return joined
