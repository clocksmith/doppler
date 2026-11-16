"""Request-owned convolution and gated-delta state; shared weights retain no private cache."""
from flash_request_state import ClientContext as DeltaContext,ClientRequestState


class DeltaRequestState(ClientRequestState):
    def __init__(self,context,layer_index=0):
        super().__init__(context,layer_index);self.conv={};self.recurrent={}

    def clear_private_state(self):self.conv.clear();self.recurrent.clear()

    def get_conv_state(self,layer_idx,state_idx=0):return self.conv.get(self._slot(layer_idx,state_idx))
    def get_recurrent_state(self,layer_idx,state_idx=0):return self.recurrent.get(self._slot(layer_idx,state_idx))
    def is_recording_past(self,layer_idx):self._slot(layer_idx);return False

    def update_conv_state(self,value,layer_idx,conv_kernel_size):
        import torch
        from torch.nn import functional as F
        slot=self._slot(layer_idx)
        self.require(type(conv_kernel_size) is int and conv_kernel_size>=2 and value.ndim==3)
        previous=self.conv.get(slot)
        joined=value if previous is None else torch.cat([previous,value],dim=-1)
        length=conv_kernel_size-1
        owned=joined[:,:,-length:].detach().clone()
        if owned.shape[-1]<length:owned=F.pad(owned,(length-owned.shape[-1],0))
        self.conv[slot]=owned
        return joined

    def update_recurrent_state(self,value,layer_idx):
        slot=self._slot(layer_idx)
        self.recurrent[slot]=value.detach().clone()
