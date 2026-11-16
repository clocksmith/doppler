"""Transaction and namespace ownership for trusted-client Flash request state."""
from dataclasses import dataclass
from protocol import require_active


@dataclass(frozen=True)
class ClientContext:
    tenant_id:str
    privacy_id:str
    model_id:str
    key_id:str
    request_id:str

    def __post_init__(self):
        if any(not isinstance(value,str) or not 0<len(value)<=256 for value in
               [self.tenant_id,self.privacy_id,self.model_id,self.key_id,self.request_id]):
            raise ValueError('Explicit tenant/privacy/model/key/request context required')


class ClientRequestState:
    def __init__(self,context,layer_index=0):
        if not isinstance(context,ClientContext) or type(layer_index) is not int or layer_index<0:
            raise ValueError('Explicit Flash request owner required')
        self._context=context;self.layer_index=layer_index
        self.failed=False;self.active=False;self.position=0
        self.cancelled=None;self.operations=();self.tokens=0;self.execution_mode=None

    @property
    def context(self):return self._context

    def fail(self):
        self.failed=True;self.active=False;self.clear_private_state()

    def require(self,condition):
        if self.failed or not condition:
            self.fail();raise ValueError('Flash request context or execution changed')
        try:require_active(self.cancelled)
        except BaseException:
            self.fail();raise

    def begin(self,context,operations,cancelled,execution):
        self.require(not self.active and context==self.context and execution in ['public-reference','verified-projections']
                     and (self.execution_mode is None or self.execution_mode==execution))
        self.execution_mode=execution
        self.cancelled=cancelled;self.require(True)
        self.active=True;self.position=0;self.operations=tuple(operations)

    def projection(self,name):
        self.require(self.active and self.position<len(self.operations) and self.operations[self.position]==name)

    def verified(self):
        self.require(self.active);self.position+=1

    def complete(self,tokens):
        self.require(self.active and self.position==len(self.operations))
        self.tokens+=tokens;self.active=False

    def _slot(self,layer_idx,state_idx=0):
        self.require(self.active and type(layer_idx) is int and layer_idx==self.layer_index and type(state_idx) is int and state_idx==0)
        return layer_idx,state_idx

    def ple_state_for(self,layer_idx):
        self.require(False)

    def clear_private_state(self):raise NotImplementedError
    def close(self):self.fail()
