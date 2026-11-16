"""Client-owned full-attention KV and QSA-indexer buffers."""
import torch
from flash_request_state import ClientRequestState


class AttentionRequestState(ClientRequestState):
    def __init__(self,context,layer_index=0):
        super().__init__(context,layer_index);self.keys={};self.values={};self.indexer={}

    def clear_private_state(self):self.keys.clear();self.values.clear();self.indexer.clear()

    def update_indexer(self,value,layer_idx):
        slot=self._slot(layer_idx);previous=self.indexer.get(slot)
        result=value if previous is None else torch.cat([previous,value],dim=1)
        self.indexer[slot]=result.detach().clone()
        return self.indexer[slot]

    def update(self,keys,values,layer_idx,**kwargs):
        slot=self._slot(layer_idx);previous_keys=self.keys.get(slot);previous_values=self.values.get(slot)
        self.require((previous_keys is None)==(previous_values is None))
        if previous_keys is not None:
            keys=torch.cat([previous_keys,keys],dim=-2);values=torch.cat([previous_values,values],dim=-2)
        self.keys[slot]=keys.detach().clone();self.values[slot]=values.detach().clone()
        return self.keys[slot],self.values[slot]
