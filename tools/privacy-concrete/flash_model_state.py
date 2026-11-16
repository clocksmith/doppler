"""One request-owned Flash cache router over existing DeltaNet/QSA/PLE state owners."""
from flash_request_state import ClientRequestState
from flash_delta_state import DeltaRequestState
from flash_attention_state import AttentionRequestState
from flash_ple_state import PLERequestState


class FlashModelState(ClientRequestState):
    def __init__(self,context,layer_types,ple_layers,model_id):
        super().__init__(context)
        if context.model_id!=model_id or any(kind not in ['linear_attention','indexed_attention'] for kind in layer_types):
            raise ValueError('Exact client reference model state required')
        self.model_id=model_id;self.layer_types=tuple(layer_types);self.ple_layers=tuple(ple_layers)
        self.attention=tuple((DeltaRequestState if kind=='linear_attention' else AttentionRequestState)(context,i) for i,kind in enumerate(layer_types))
        self.ple={i:PLERequestState(context,i) for i in ple_layers}

    def children(self):return (*self.attention,*self.ple.values())

    def clear_private_state(self):
        for state in self.children():state.close()

    def begin(self,context,operations,cancelled,execution):
        super().begin(context,operations,cancelled,execution)
        try:
            for state in self.children():
                self.require(state.tokens==self.tokens)
                state.begin(context,(),cancelled,execution)
        except BaseException:self.fail();raise

    def complete(self,tokens):
        try:
            for state in self.children():state.complete(tokens)
            super().complete(tokens)
        except BaseException:self.fail();raise

    def _attention(self,layer_idx,kind):
        self.require(self.active and type(layer_idx) is int and 0<=layer_idx<len(self.attention) and self.layer_types[layer_idx]==kind)
        return self.attention[layer_idx]

    def ple_state_for(self,layer_idx):
        self.require(self.active and type(layer_idx) is int and layer_idx in self.ple)
        return self.ple[layer_idx]

    def _conv(self,layer_idx,state_idx):
        self.require(type(state_idx) is int and state_idx in [0,1,2])
        if state_idx==0:return self._attention(layer_idx,'linear_attention')
        self.require(type(state_idx) is int and state_idx in [1,2])
        return self.ple_state_for(layer_idx)

    def get_conv_state(self,layer_idx,state_idx=0):return self._conv(layer_idx,state_idx).get_conv_state(layer_idx,state_idx)
    def update_conv_state(self,value,layer_idx,state_idx=0,conv_kernel_size=None):
        state=self._conv(layer_idx,state_idx)
        if state_idx==0:return state.update_conv_state(value,layer_idx,conv_kernel_size)
        return state.update_conv_state(value,layer_idx,state_idx,conv_kernel_size)
    def get_recurrent_state(self,layer_idx,state_idx=0):return self._attention(layer_idx,'linear_attention').get_recurrent_state(layer_idx,state_idx)
    def update_recurrent_state(self,value,layer_idx):return self._attention(layer_idx,'linear_attention').update_recurrent_state(value,layer_idx)
    def is_recording_past(self,layer_idx):return self._attention(layer_idx,'linear_attention').is_recording_past(layer_idx)
    def update(self,keys,values,layer_idx,**kwargs):return self._attention(layer_idx,'indexed_attention').update(keys,values,layer_idx,**kwargs)
    def update_indexer(self,value,layer_idx):return self._attention(layer_idx,'indexed_attention').update_indexer(value,layer_idx)
