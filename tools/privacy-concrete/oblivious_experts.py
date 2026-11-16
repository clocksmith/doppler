"""Trusted-client all-expert SwiGLU composition; a fixed access schedule, not a crypto backend."""
from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from pathlib import Path
from full_plan import canonical
from protocol import digest, require_active


def vector(value,width):
    if (not isinstance(value,(list,tuple)) or len(value)!=width
            or any(type(item) not in (int,float) or not math.isfinite(item) for item in value)):
        raise ValueError('Expert operator requires finite explicit vector geometry')
    return value


def silu(value):
    if value>=0:return value/(1+math.exp(-value))
    exp=math.exp(value)
    return value*exp/(1+exp)


@dataclass(frozen=True)
class ExpertProgram:
    source_config_id: str
    expert_count: int
    top_k: int
    hidden_width: int
    intermediate_width: int

    def __post_init__(self):
        if (not isinstance(self.source_config_id,str) or len(self.source_config_id)!=64
                or any(char not in '0123456789abcdef' for char in self.source_config_id)
                or any(type(value) is not int or value<=0 for value in
                       [self.expert_count,self.top_k,self.hidden_width,self.intermediate_width])
                or self.top_k>self.expert_count):
            raise ValueError('Explicit source and expert geometry required')

    def contract(self):
        return dict(schema='doppler.client-all-expert-swiglu/v1',sourceConfigId=self.source_config_id,
                    expertCount=self.expert_count,topK=self.top_k,hiddenWidth=self.hidden_width,
                    intermediateWidth=self.intermediate_width,
                    operations=[dict(id=f'expert/{expert}/{name}',inputWidth=inputs,outputWidth=outputs)
                                for expert in range(self.expert_count)
                                for name,inputs,outputs in [('gate',self.hidden_width,self.intermediate_width),
                                                          ('up',self.hidden_width,self.intermediate_width),
                                                          ('down',self.intermediate_width,self.hidden_width)]])

    @property
    def identity(self):return digest(canonical(self.contract()))

    def attempt(self):return ExpertAttempt(self)


def flash_expert_program():
    source=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())
    config=source['config']
    if config['hidden_act']!='silu':raise ValueError('Unsupported expert activation')
    return ExpertProgram(source['sourceConfigSha256'],config['num_experts'],config['num_experts_per_tok'],
                         config['hidden_size'],config['moe_intermediate_size'])


class ExpertAttempt:
    def __init__(self,program):
        self.program,self.position,self.failed=program,0,False

    def run(self,hidden,selected_ids,routing_weights,verified_client_projection,cancelled=None):
        """Projection callback is a trusted client adapter returning already-verified local plaintext.

        No network service may be passed directly. This routine does not implement encryption,
        authenticate an evaluator, appraise a backend or qualify Flash model semantics.
        """
        if self.failed or self.position:
            raise ValueError('Expert attempts cannot retry or reuse private state')
        try:
            spec=self.program
            hidden=tuple(vector(hidden,spec.hidden_width))
            routing_weights=tuple(vector(routing_weights,spec.top_k))
            if (not isinstance(selected_ids,(list,tuple)) or len(selected_ids)!=spec.top_k
                    or any(type(index) is not int or not 0<=index<spec.expert_count for index in selected_ids)
                    or len(set(selected_ids))!=len(selected_ids)):
                raise ValueError('Private expert selection does not match the declared geometry')
            selected_ids=tuple(selected_ids)
            operations=tuple(MappingProxyType(operation) for operation in spec.contract()['operations'])
            def project(value):
                require_active(cancelled)
                operation=operations[self.position]
                owned=tuple(vector(value,operation['inputWidth']))
                result=verified_client_projection(operation,owned)
                require_active(cancelled)
                owned_result=tuple(vector(result,operation['outputWidth']))
                self.position+=1
                return owned_result
            outputs=[]
            for _expert in range(spec.expert_count):
                gate=project(hidden)
                up=project(hidden)
                activated=vector([silu(left)*right for left,right in zip(gate,up)],spec.intermediate_width)
                outputs.append(project(activated))
            if self.position!=len(operations):raise ValueError('Incomplete all-expert execution')
            # Combine only after every expert has passed verification. No selected-only
            # remote calls, sparse weight reads or selected-only integrity checks.
            result=[sum(weight*outputs[index][channel] for index,weight in zip(selected_ids,routing_weights))
                    for channel in range(spec.hidden_width)]
            return list(vector(result,spec.hidden_width))
        except BaseException:
            self.failed=True
            raise


def torch_router(logits,top_k,normalize):
    """Match the pinned source's float32 softmax/topk on the trusted client, including Torch ties."""
    import torch
    values=torch.tensor(logits,dtype=torch.float32)
    weights,indices=torch.softmax(values,dim=-1,dtype=torch.float32).topk(top_k,dim=-1)
    if normalize:weights=weights/weights.sum()
    return indices.tolist(),weights.tolist()


@dataclass(frozen=True)
class FlashMoEProgram:
    experts: ExpertProgram
    shared_width: int
    normalize_topk: bool
    reference_id: str

    def __post_init__(self):
        if (type(self.shared_width) is not int or self.shared_width<=0 or type(self.normalize_topk) is not bool
                or not isinstance(self.reference_id,str) or len(self.reference_id)!=64
                or any(char not in '0123456789abcdef' for char in self.reference_id)):
            raise ValueError('Explicit Flash shared-expert and reference semantics required')

    def contract(self):
        hidden=self.experts.hidden_width;shared=self.shared_width
        def operation(name,inputs,outputs):return dict(id=name,inputWidth=inputs,outputWidth=outputs)
        operations=[operation('shared/gate',hidden,shared),operation('shared/up',hidden,shared),
                    operation('shared/down',shared,hidden),operation('router/logits',hidden,self.experts.expert_count)]
        operations+=self.experts.contract()['operations']
        operations.append(operation('shared/mix',hidden,1))
        return dict(schema='doppler.client-flash-moe-reference/v1',sourceConfigId=self.experts.source_config_id,expertProgramId=self.experts.identity,
                    operatorReferenceId=self.reference_id,normalizeTopK=self.normalize_topk,rows=1,operations=operations)

    @property
    def identity(self):return digest(canonical(self.contract()))
    def attempt(self):return FlashMoEAttempt(self)


def flash_moe_program():
    source=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())
    return FlashMoEProgram(flash_expert_program(),source['config']['shared_expert_intermediate_size'],
                           source['resolvedOperatorDefaults']['normTopKProbability'],digest(canonical(source['operatorReference'])))


class FlashMoEAttempt:
    def __init__(self,program):self.program,self.position,self.failed=program,0,False

    def run(self,hidden,verified_client_projection,cancelled=None):
        if self.failed or self.position:raise ValueError('Flash operator attempts cannot retry or reuse private state')
        try:
            program=self.program;spec=program.experts
            hidden=tuple(vector(hidden,spec.hidden_width))
            operations=tuple(MappingProxyType(operation) for operation in program.contract()['operations'])
            def project(value,expected=None):
                require_active(cancelled)
                operation=operations[self.position]
                if expected is not None and dict(expected)!=dict(operation):
                    raise ValueError('Complete Flash expert operator order changed')
                value=tuple(vector(value,operation['inputWidth']))
                result=verified_client_projection(operation,value)
                require_active(cancelled)
                result=tuple(vector(result,operation['outputWidth']))
                self.position+=1
                return result
            gate=project(hidden);up=project(hidden)
            shared=project([silu(left)*right for left,right in zip(gate,up)])
            logits=project(hidden)
            selected,weights=torch_router(logits,spec.top_k,program.normalize_topk)
            routed=spec.attempt().run(hidden,selected,weights,lambda operation,value:project(value,operation),cancelled)
            mix=project(hidden)[0]
            if self.position!=len(operations):raise ValueError('Incomplete Flash expert operator program')
            # sigmoid uses the same stable sign split as SiLU; avoid dividing by mix at zero.
            factor=1/(1+math.exp(-mix)) if mix>=0 else math.exp(mix)/(1+math.exp(mix))
            return list(vector([left+factor*right for left,right in zip(routed,shared)],spec.hidden_width))
        except BaseException:
            self.failed=True
            raise
