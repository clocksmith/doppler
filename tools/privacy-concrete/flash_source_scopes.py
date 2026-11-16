"""Installed public source scopes; metadata advertisements cannot add tensor or shard paths."""
import copy
import json
from pathlib import Path

BASE=json.loads(Path(__file__).with_name('flash-weights.json').read_text())
HYPER=json.loads(Path(__file__).with_name('flash-hyper-tensors.json').read_text())
PLE=json.loads(Path(__file__).with_name('flash-ple-weights.json').read_text())
QSA=json.loads(Path(__file__).with_name('flash-qsa-weights.json').read_text())
DELTA=json.loads(Path(__file__).with_name('flash-delta-weights.json').read_text())
EXPERT_TENSORS=frozenset(name for name in BASE['tensors'] if '.experts.' in name)
SCOPES=('moe','mlp-hyper','mlp-block','linear-attention','full-attention','ple')


def source_spec(scope):
    if scope not in SCOPES:raise ValueError('Unsupported installed Flash source scope')
    result=copy.deepcopy(BASE)
    if scope=='moe':return result
    result['schema']='doppler.flash-component-weights/v1'
    if scope in ['linear-attention','full-attention','ple']:
        selected={'linear-attention':DELTA,'full-attention':QSA,'ple':PLE}[scope]
        result['shards']=copy.deepcopy(selected['shards']);result['tensors']=copy.deepcopy(selected['tensors'])
        return result
    if scope=='mlp-hyper':result['tensors']={}
    result['tensors'].update(copy.deepcopy(HYPER))
    required={row['shard'] for row in result['tensors'].values()}
    result['shards']={name:record for name,record in result['shards'].items() if name in required}
    return result
