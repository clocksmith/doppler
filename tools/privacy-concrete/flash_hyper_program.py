"""Pure source-bound Flash hyperconnection program and geometry."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from full_plan import canonical
from protocol import digest

@dataclass(frozen=True)
class HyperProgram:
    source_config_id:str
    reference_id:str
    streams:int
    hidden_width:int
    low_rank:int
    epsilon:float

    def __post_init__(self):
        if (any(not isinstance(value,str) or len(value)!=64 or any(char not in '0123456789abcdef' for char in value)
                for value in [self.source_config_id,self.reference_id])
                or any(type(value) is not int or value<=0 for value in [self.streams,self.hidden_width,self.low_rank])
                or type(self.epsilon) is not float or not math.isfinite(self.epsilon) or self.epsilon<=0):
            raise ValueError('Explicit Flash hyperconnection source and geometry required')

    def contract(self):
        width=self.streams*self.hidden_width
        return dict(schema='doppler.flash-client-hyperconnection/v1',sourceConfigId=self.source_config_id,
                    operatorReferenceId=self.reference_id,streams=self.streams,hiddenWidth=self.hidden_width,
                    lowRank=self.low_rank,epsilon=self.epsilon,
                    operations=[dict(id='hyper/down',inputWidth=width,outputWidth=self.low_rank),
                                dict(id='hyper/up',inputWidth=self.low_rank,outputWidth=width),
                                dict(id='hyper/inject',inputWidth=width,outputWidth=self.streams)])

    @property
    def identity(self):return digest(canonical(self.contract()))



def flash_hyper_program():
    config=json.loads(Path(__file__).with_name('flash-hyper-program.json').read_text())
    source=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())
    return HyperProgram(source['sourceConfigSha256'],digest(canonical(source['operatorReference'])),
                        config['streams'],config['hiddenWidth'],config['lowRank'],config['epsilon'])


def require_source_config(program,config):
    expected=dict(hc_count=program.streams,hidden_size=program.hidden_width,hc_lowrank=program.low_rank,rms_norm_eps=program.epsilon)
    if any(canonical(config[key])!=canonical(value) for key,value in expected.items()):
        raise ValueError('Pinned Flash hyperconnection source geometry changed')
