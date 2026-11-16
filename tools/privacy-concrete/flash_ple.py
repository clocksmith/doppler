"""Local-materialized ngram/PLE client reference; no remote lookup or actual Flash table support."""
import json
from pathlib import Path
import torch
from torch import nn
from flash_reference import load_reference
from flash_ple_state import PLERequestState
from flash_client_projections import ClientProjection,client_execution

CONFIG=json.loads(Path(__file__).with_name('flash-ple-program.json').read_text());ORDER=tuple(CONFIG['projectionOrder'])


class ClientPLE(nn.Module):
    def __init__(self,reference_path,config,*,context_limit,embedding_limit_bytes,layer_index=1,ple_layer_index=0):
        super().__init__()
        if (type(context_limit) is not int or not 0<context_limit<=config.max_position_embeddings
                or type(embedding_limit_bytes) is not int or embedding_limit_bytes<=0):
            raise ValueError('Explicit bounded local PLE context and embedding budget required')
        source=load_reference(reference_path,'ple')
        if tuple(CONFIG['sourceTransforms'])!=source['__referenceTransforms__']:raise ValueError('PLE source transform contract changed')
        cls=source['Qwen4ExpTextPLELayer']
        with torch.device('meta'):candidate=cls(config,layer_index,ple_layer_index)
        table=candidate.ple_embedding.ngram_embedding.weight
        if table.numel()*table.element_size()>embedding_limit_bytes:
            raise ValueError('Complete local ngram table exceeds the declared materialization budget')
        del candidate
        self.model=cls(config,layer_index,ple_layer_index).eval().requires_grad_(False)
        self.layer_index=layer_index;self.context_limit=context_limit;self.hidden_width=config.hidden_size*config.hc_count
        self.vocabulary_size=config.vocab_size;self.reference_transforms=source['__referenceTransforms__']
        for name in ORDER:setattr(self.model,name,ClientProjection(name,getattr(self.model,name)))

    def forward(self,value,input_ids,state,context,*,execution,verified_client_projection=None,cancelled=None):
        if not isinstance(state,PLERequestState):raise ValueError('Request-owned PLE state required')
        with client_execution(state,context,ORDER,execution,verified_client_projection,cancelled):
            state.require(state.layer_index==self.layer_index and value.dtype==torch.float32 and value.ndim==3
                          and value.shape[0]==1 and value.shape[-1]==self.hidden_width and value.shape[1]>0
                          and tuple(input_ids.shape)==tuple(value.shape[:2]) and input_ids.dtype==torch.int64
                          and torch.isfinite(value).all() and (input_ids>=0).all() and (input_ids<self.vocabulary_size).all()
                          and state.tokens+value.shape[1]<=self.context_limit
                          and type(self.model.ple_embedding.ngram_embedding) is nn.Embedding)
            with torch.no_grad():result=self.model(value.clone(),input_ids.clone(),cache=state)
            state.require(tuple(result.shape)==tuple(value.shape) and torch.isfinite(result).all())
            state.complete(value.shape[1]);return result
