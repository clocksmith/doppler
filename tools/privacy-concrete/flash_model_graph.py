"""Complete client Flash reference assembly over trusted materialized source components.

Includes embeddings, every decoder, dense all-expert routing, final stream mixer,
vocabulary head and request-local autoregressive cache routing. This is a reference
composition, not an actual-weight model loader, encrypted backend or qualification.
Callbacks are trusted client code, never directly a plaintext network service.
"""
import hashlib
import torch
from torch import nn
from full_plan import canonical
from protocol import digest
from flash_client_projections import ClientProjection,client_execution
from flash_model_state import FlashModelState
from flash_delta_program import ORDER as DELTA_ORDER
from flash_attention import ORDER as QSA_ORDER
from flash_ple import ORDER as PLE_ORDER
from flash_rotary import text_rotary
from flash_reference import SOURCE,SPEC

HYPER=('input_mix_weight_down','input_mix_weight_up','block_inject_weight')


class DenseClientMoE(nn.Module):
    def __init__(self,source,config):
        super().__init__();self.config=config;self.ops=nn.ModuleDict()
        def add(name,weight):
            with torch.device('meta'):layer=nn.Linear(weight.shape[1],weight.shape[0],bias=False)
            layer.weight=nn.Parameter(weight.detach(),requires_grad=False);self.ops[name]=layer
        for name in ['gate','up','down']:add('shared_'+name,getattr(source.shared_expert,name+'_proj').weight)
        add('router',source.gate.weight)
        for index in range(config.num_experts):
            packed=source.experts.gate_up_proj[index];width=config.moe_intermediate_size
            add(f'expert_{index}_gate',packed[:width]);add(f'expert_{index}_up',packed[width:])
            add(f'expert_{index}_down',source.experts.down_proj[index])
        add('shared_mix',source.shared_expert_gate.weight)

    def forward(self,value):
        ops=self.ops
        shared=ops['shared_down'](torch.nn.functional.silu(ops['shared_gate'](value))*ops['shared_up'](value))
        probabilities=torch.softmax(ops['router'](value),dim=-1,dtype=torch.float32)
        weights,indices=probabilities.topk(self.config.num_experts_per_tok,dim=-1)
        if self.config.norm_topk_prob:weights=weights/weights.sum(-1,keepdim=True)
        all_outputs=[]
        for i in range(self.config.num_experts):
            all_outputs.append(ops[f'expert_{i}_down'](torch.nn.functional.silu(ops[f'expert_{i}_gate'](value))*ops[f'expert_{i}_up'](value)))
        all_outputs=torch.stack(all_outputs,dim=-2)
        routing=torch.zeros_like(probabilities).scatter(-1,indices,weights)
        routed=(all_outputs*routing.unsqueeze(-1)).sum(-2)
        return routed+torch.sigmoid(ops['shared_mix'](value))*shared


class ClientFlashGraph(nn.Module):
    def __init__(self,reference_path,config,embedding,layers,final_mixer,head,*,context_limit):
        super().__init__()
        if (type(context_limit) is not int or not 0<context_limit<=config.max_position_embeddings
                or len(layers)!=config.num_hidden_layers or len(config.layer_types)!=len(layers)
                or any(kind not in ['linear_attention','indexed_attention'] for kind in config.layer_types)
                or config._attn_implementation!='eager' or config.rope_parameters['rope_type']!='default'):
            raise ValueError('Complete bounded inspected client model components required')
        self.reference_path=reference_path;self.config=config;self.context_limit=context_limit
        self.embedding=embedding;self.layers=nn.ModuleList(layers);self.final_mixer=final_mixer;self.head=head
        self.ple_layers=tuple(i for i in range(len(layers)) if i+1 in config.ple_layer_ids)
        order=[]
        for i,layer in enumerate(self.layers):
            if layer.layer_type!=config.layer_types[i] or (layer.ple is not None)!=(i in self.ple_layers):raise ValueError('Client decoder/PLE route changed')
            prefix=f'layers.{i}.'
            if layer.ple is not None:order.extend(prefix+'ple.'+name for name in PLE_ORDER)
            order.extend(prefix+'attn_hyper_connection.'+name for name in HYPER)
            kind='linear_attn' if layer.layer_type=='linear_attention' else 'self_attn'
            order.extend(prefix+kind+'.'+name for name in (DELTA_ORDER if kind=='linear_attn' else QSA_ORDER))
            order.extend(prefix+'mlp_hyper_connection.'+name for name in HYPER)
            layer.mlp=DenseClientMoE(layer.mlp,config)
            order.extend(prefix+'mlp.ops.'+name for name in layer.mlp.ops)
        order.extend('final_mixer.'+name for name in HYPER[:2]);order.append('head')
        public={}
        for name,layer in list(self.named_modules()):
            if isinstance(layer,nn.Linear):
                parent,_,leaf=name.rpartition('.');owner=self.get_submodule(parent) if parent else self
                wrapped=ClientProjection(name,layer);setattr(owner,leaf,wrapped);public[name]=wrapped
        if set(public)!=set(order):raise ValueError('Unaccounted client model projection')
        self.order=tuple(order);self.projections=public
        hasher=hashlib.sha256()
        hasher.update(canonical(dict(sourceReferenceId=digest(canonical(SOURCE)),sourceTransforms=SPEC['transforms']['decoder'],layerTypes=config.layer_types,pleLayers=list(self.ple_layers),contextLimit=context_limit,operations=[dict(public[name].operation) for name in order])))
        for name,value in self.state_dict().items():
            hasher.update(canonical(dict(name=name,dtype=str(value.dtype),shape=list(value.shape))));hasher.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.identity=hasher.hexdigest();self.eval().requires_grad_(False)

    def new_state(self,context):return FlashModelState(context,self.config.layer_types,self.ple_layers,self.identity)

    def forward(self,tokens,state,context,*,execution,verified_client_projection=None,cancelled=None):
        if type(state) is not FlashModelState:raise ValueError('Request-owned complete Flash state required')
        with client_execution(state,context,self.order,execution,verified_client_projection,cancelled):
            state.require(state.model_id==self.identity and tokens.dtype==torch.int64 and tokens.device.type=='cpu'
                          and tokens.ndim==2 and tokens.shape[0]==1 and tokens.shape[1]>0
                          and (tokens>=0).all() and (tokens<self.config.vocab_size).all() and state.tokens+tokens.shape[1]<=self.context_limit)
            total=state.tokens+tokens.shape[1];past=state.tokens
            positions=text_rotary(self.reference_path,self.config,total)
            visible=torch.arange(total)[None,:]<=torch.arange(past,total)[:,None]
            mask=torch.full((1,1,total-past,total),torch.finfo(torch.float32).min).masked_fill(visible[None,None],0)
            with torch.no_grad():
                hidden=self.embedding(tokens.clone()).repeat(1,1,self.config.hc_count)
                for layer in self.layers:hidden=layer(hidden,position_embeddings=positions,attention_mask=mask,conv_mask=None,cache=state,ple_input_ids=tokens.clone())
                logits=self.head(self.final_mixer(hidden))
            state.require(torch.isfinite(logits).all());state.complete(tokens.shape[1]);return logits.detach().clone()
