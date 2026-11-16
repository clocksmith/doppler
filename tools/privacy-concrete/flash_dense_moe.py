"""Fixed all-expert client computation shared by materialized and source-bound graphs."""
import torch
from torch import nn


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


