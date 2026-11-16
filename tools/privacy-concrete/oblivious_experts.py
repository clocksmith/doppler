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
