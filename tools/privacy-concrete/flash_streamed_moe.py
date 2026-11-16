"""Source-bound all-expert client MoE; one public projection is materialized per call.

No downloads, private routing to a provider, encrypted admission or full-model
qualification. The caller owns the already verified public decoder-stage lease.
"""
from pathlib import Path
from types import SimpleNamespace,MappingProxyType
import torch
from torch import nn
from flash_dense_moe import DenseClientMoE
from flash_source_scopes import staging_spec
from flash_weights import FlashSource
from full_plan import canonical
from protocol import digest
from source_snapshot import file_signature


class ProjectionSource:
    """One public source snapshot shared across its projections; no request state."""
    def __init__(self,source):
        if not isinstance(source,FlashSource):raise ValueError('Verified public projection source required')
        self.source=source;self.directory=Path(source.directory)
        self.snapshot=canonical(source.spec);self.identity=digest(self.snapshot)
        self.records=MappingProxyType({name:canonical(row) for name,row in source.spec['tensors'].items()})
        self.shards=MappingProxyType({name:row['shard'] for name,row in source.spec['tensors'].items()})
        self.signatures=MappingProxyType(dict(source.signatures))
        self.require_source()

    def require_source(self):
        if (getattr(self.source,'closed',False) or Path(self.source.directory)!=self.directory
                or canonical(self.source.spec)!=self.snapshot):
            raise ValueError('Verified public source changed or stage closed')
        for shard in set(self.shards.values()):self.require_file(shard)

    def require_file(self,shard):
        if (self.source.signatures.get(shard)!=self.signatures[shard]
                or file_signature(self.directory/shard)!=self.signatures[shard]):
            raise ValueError('Verified public projection file changed')

    def require_tensor(self,name):
        if (getattr(self.source,'closed',False) or Path(self.source.directory)!=self.directory
                or canonical(self.source.spec['tensors'].get(name))!=self.records[name]):
            raise ValueError('Verified public projection source changed or stage closed')
        self.require_file(self.shards[name])


class SourceLinear(nn.Linear):
    """Meta geometry plus immutable source binding; never retain a floating weight."""
    def __init__(self,source,name,input_width,output_width,*,expert=None,rows=None,source_owner=None):
        if not isinstance(source,FlashSource):raise ValueError('Verified public projection source required')
        record=source.spec['tensors'][name];shape=record['shape']
        if expert is not None:
            if len(shape)!=3 or type(expert) is not int or not 0<=expert<shape[0]:raise ValueError('Explicit expert projection required')
            shape=shape[1:]
        if len(shape)!=2 or record['dtype']!='BF16':raise ValueError('Public BF16 matrix required')
        if rows is not None:
            if (type(rows) is not tuple or len(rows)!=2 or any(type(value) is not int for value in rows)
                    or not 0<=rows[0]<rows[1]<=shape[0]):raise ValueError('Bound public projection rows required')
            shape=[rows[1]-rows[0],shape[1]]
        if (type(input_width) is not int or type(output_width) is not int or (output_width,input_width)!=tuple(shape)):
            raise ValueError('Exact public projection geometry required')
        with torch.device('meta'):super().__init__(input_width,output_width,bias=False)
        self._source=source;self._name=name;self._expert=expert;self._rows=rows
        self.source_owner=ProjectionSource(source) if source_owner is None else source_owner
        if type(self.source_owner) is not ProjectionSource or self.source_owner.source is not source:
            raise ValueError('Matching verified public source owner required')
        self._source_binding=canonical(dict(sourceId=self.source_owner.identity,tensor=name,expert=expert,
            rows=rows,inputWidth=input_width,outputWidth=output_width))
        self.require_source()

    @property
    def source_binding(self):return self._source_binding

    def require_source(self):self.source_owner.require_tensor(self._name)

    def forward(self,value):
        self.require_source()
        weight=self._source.tensor(self._name,expert=self._expert)
        if self._rows is not None:weight=weight[self._rows[0]:self._rows[1]]
        if (weight.device.type!='cpu' or weight.dtype!=torch.float32 or weight.shape!=(self.out_features,self.in_features)
                or not torch.isfinite(weight).all()):raise ValueError('Bound public projection materialization changed')
        # No autograd graph may retain this public matrix between projections.
        with torch.no_grad():result=torch.nn.functional.linear(value,weight)
        self.require_source()
        return result


class SourceDenseClientMoE(DenseClientMoE):
    def __init__(self,source,config,layer_index):
        nn.Module.__init__(self)
        if not isinstance(source,FlashSource) or canonical(source.spec)!=canonical(staging_spec('decoder',layer_index)):
            raise ValueError('Exact verified public decoder-stage source required')
        if type(config) is not SimpleNamespace:raise ValueError('Explicit client MoE configuration required')
        names=('hidden_size','num_experts','num_experts_per_tok','moe_intermediate_size','shared_expert_intermediate_size')
        if (any(type(getattr(config,name,None)) is not int or getattr(config,name)<=0 for name in names)
                or config.num_experts_per_tok>config.num_experts or type(getattr(config,'norm_topk_prob',None)) is not bool):
            raise ValueError('Bound client MoE geometry required')
        self.source_owner=ProjectionSource(source)
        self.config=config;self._config_snapshot=canonical(vars(config));self.ops=nn.ModuleDict()
        prefix=f'model.language_model.layers.{layer_index}.mlp.'
        hidden,inner,shared=config.hidden_size,config.moe_intermediate_size,config.shared_expert_intermediate_size
        def add(operation,tensor,shape,*,expert=None,rows=None):
            row=source.spec['tensors'].get(prefix+tensor)
            full_shape=([config.num_experts,*shape] if expert is not None else shape)
            if not isinstance(row,dict) or row.get('shape')!=full_shape or row.get('dtype')!='BF16':
                raise ValueError('Pinned public MoE projection geometry required')
            height=shape[0] if rows is None else rows[1]-rows[0]
            self.ops[operation]=SourceLinear(source,prefix+tensor,shape[1],height,expert=expert,rows=rows,source_owner=self.source_owner)
        add('shared_gate','shared_expert.gate_proj.weight',[shared,hidden])
        add('shared_up','shared_expert.up_proj.weight',[shared,hidden])
        add('shared_down','shared_expert.down_proj.weight',[hidden,shared])
        add('router','gate.weight',[config.num_experts,hidden])
        for index in range(config.num_experts):
            add(f'expert_{index}_gate','experts.gate_up_proj',[2*inner,hidden],expert=index,rows=(0,inner))
            add(f'expert_{index}_up','experts.gate_up_proj',[2*inner,hidden],expert=index,rows=(inner,2*inner))
            add(f'expert_{index}_down','experts.down_proj',[hidden,inner],expert=index)
        add('shared_mix','shared_expert_gate.weight',[1,hidden])
        self.eval().requires_grad_(False)

    def forward(self,value):
        self.source_owner.require_source()
        if canonical(vars(self.config))!=self._config_snapshot:raise ValueError('Client MoE configuration changed')
        return super().forward(value)
