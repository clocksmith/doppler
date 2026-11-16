"""Installed public source scopes; metadata advertisements cannot add tensor or shard paths."""
import copy
import json
from pathlib import Path
from full_plan import canonical
from protocol import digest

BASE=json.loads(Path(__file__).with_name('flash-weights.json').read_text())
HYPER=json.loads(Path(__file__).with_name('flash-hyper-tensors.json').read_text())
PLE=json.loads(Path(__file__).with_name('flash-ple-weights.json').read_text())
QSA=json.loads(Path(__file__).with_name('flash-qsa-weights.json').read_text())
DELTA=json.loads(Path(__file__).with_name('flash-delta-weights.json').read_text())
TEXT=json.loads(Path(__file__).with_name('flash-text-weights.json').read_text())
EXPERT_TENSORS=frozenset(name for specification in [BASE,TEXT] for name in specification['tensors'] if '.mlp.experts.' in name)
SCOPES=('moe','mlp-hyper','mlp-block','linear-attention','full-attention','ple','text')


def source_spec(scope):
    if scope not in SCOPES:raise ValueError('Unsupported installed Flash source scope')
    if scope=='text':return copy.deepcopy(TEXT)
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


def staging_spec(part,layer_index=None):
    """Explicit public materialization subset, never an executable full-model offer."""
    table=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())
    if part=='decoder':
        if type(layer_index) is not int or not 0<=layer_index<TEXT['layerCount']:
            raise ValueError('Explicit installed decoder layer required')
        prefix=f'model.language_model.layers.{layer_index}.'
        selected={name:row for name,row in TEXT['tensors'].items() if name.startswith(prefix)}
        separate={row['name'] for row in table['parts']}
        separate_names=sorted(set(selected)&separate)
        selected={name:row for name,row in selected.items() if name not in separate}
    elif part=='output' and layer_index is None:
        separate_names=[]
        selected={name:row for name,row in TEXT['tensors'].items() if name=='lm_head.weight' or name.startswith('model.language_model.hyper_connection_mixer.')}
    else:raise ValueError('Installed public model-part staging required')
    if not selected:raise ValueError('Complete installed part tensor set required')
    result=copy.deepcopy(TEXT);required={row['shard'] for row in selected.values()}
    result.update(schema='doppler.flash-staged-model-part-weights/v1',part=part,layerIndex=layer_index,
                  parentSourceId=digest(canonical(TEXT)),localTableId=digest(canonical(table)),completeTextModel=False,
                  separatelyOwnedLocalTableTensors=separate_names,
                  tensors=copy.deepcopy(selected),shards={name:copy.deepcopy(TEXT['shards'][name]) for name in sorted(required)},
                  tensorCount=len(selected),layerCount=1 if part=='decoder' else 0)
    result['logicalTensorBytes']=sum((row['dataOffsets'][1]-row['dataOffsets'][0]) for row in selected.values())
    result['wholeShardBytes']=sum(row['bytes'] for row in result['shards'].values())
    result['wholeSourceBytes']=result['wholeShardBytes']+result['auxiliarySourceBytes']
    return result
