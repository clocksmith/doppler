"""Installed full-text Flash research plan; no production or numerical qualification.

The route is derived from the shipped authenticated tensor inventory, not from
provider metadata. All experts execute. Local nonlinear/table state remains on
the client, and the existing FullBridge mandates encrypted replay of every tile.
"""
import json
import stat
from pathlib import Path
from full_plan import FullPlan,canonical
from flash_source_scopes import TEXT
from flash_reference import SOURCE,SPEC
from flash_component_plan import component_versions,validate_component,write_component
from protocol import digest
from public_weights import validate_public_projection
from source_snapshot import file_signature

CONFIG=dict(schema='doppler.flash-text-research-policy/v1',profile='flash-text-all-linear-research',
    planSchema='doppler.flash-text-research-plan/v1',protocol='concrete-glwe-flash-text-replay/v1',
    productionBackendAvailable=False,steps=4,contextBucket=1,inputTileWidth=2048,outputTileWidth=2048,
    cpuThreads=4,operationTimeoutSeconds=900,additionalDependencies={'safetensors':'0.8.0'})
HYPER=('input_mix_weight_down','input_mix_weight_up','block_inject_weight')
DELTA=tuple(json.loads(Path(__file__).with_name('flash-delta-policy.json').read_text())['projectionOrder'])
QSA=tuple(json.loads(Path(__file__).with_name('flash-attention-program.json').read_text())['projectionOrder'])
PLE=tuple(json.loads(Path(__file__).with_name('flash-ple-program.json').read_text())['projectionOrder'])


def geometry(source=TEXT):
    result={};tensors=source['tensors']
    def add(operation,tensor,expert=None,half=None):
        row=tensors[tensor];shape=row['shape']
        if row['dtype']!='BF16':raise ValueError('Installed Flash BF16 projection required')
        if expert is not None:
            if len(shape)!=3 or shape[0]!=source['expertsPerLayer'] or not 0<=expert<shape[0]:raise ValueError('Installed expert geometry required')
            shape=shape[1:]
        if len(shape)!=2:raise ValueError('Installed Flash matrix required')
        height,width=shape
        if half is not None:
            if height%2:raise ValueError('Packed expert gate/up geometry required')
            height//=2
        if operation in result:raise ValueError('Duplicate installed Flash operation')
        result[operation]=(width,height)
    for i in range(source['layerCount']):
        local=f'layers.{i}.';prefix='model.language_model.'+local
        if prefix+'ple.key_proj.weight' in tensors:
            for name in PLE:add(local+'ple.'+name,prefix+'ple.'+name+'.weight')
        for name in HYPER:add(local+'attn_hyper_connection.'+name,prefix+'attn_hyper_connection.'+name+'.weight')
        kind='linear_attn' if prefix+'linear_attn.in_proj_qkv.weight' in tensors else 'self_attn'
        for name in DELTA if kind=='linear_attn' else QSA:add(local+kind+'.'+name,prefix+kind+'.'+name+'.weight')
        for name in HYPER:add(local+'mlp_hyper_connection.'+name,prefix+'mlp_hyper_connection.'+name+'.weight')
        ops=local+'mlp.ops.';mlp=prefix+'mlp.'
        for name in ['gate','up','down']:add(ops+'shared_'+name,mlp+'shared_expert.'+name+'_proj.weight')
        add(ops+'router',mlp+'gate.weight')
        for expert in range(source['expertsPerLayer']):
            for half,name in enumerate(['gate','up']):add(ops+f'expert_{expert}_'+name,mlp+'experts.gate_up_proj',expert,half)
            add(ops+f'expert_{expert}_down',mlp+'experts.down_proj',expert)
        add(ops+'shared_mix',mlp+'shared_expert_gate.weight')
    for name in HYPER[:2]:add('final_mixer.'+name,'model.language_model.hyper_connection_mixer.'+name+'.weight')
    add('head','lm_head.weight')
    return result

GEOMETRY=geometry()
PROGRAM=dict(schema='doppler.flash-text-client-program/v1',sourceId=digest(canonical(TEXT)),
    operatorReferenceId=digest(canonical(SOURCE)),sourceTransforms=SPEC['transforms']['decoder'],
    projectionOrder=list(GEOMETRY),expertRouting='fixed-all-expert',integrity='full-encrypted-tile-replay',
    clientState='nonlinear-attention-cache-tokenization-and-local-tables',productionBackendAvailable=False,securityQualified=False)
PROGRAM_ID=digest(canonical(PROGRAM))


def semantics(graph_id):
    if not isinstance(graph_id,str) or len(graph_id)!=64 or any(c not in '0123456789abcdef' for c in graph_id):raise ValueError('Explicit client graph identity required')
    return dict(programId=PROGRAM_ID,clientGraphId=graph_id,securityQualified=False,fullFlashModelQualified=False)


def versions():return component_versions(CONFIG)


class FlashTextPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    value=json.loads((Path(directory)/'plan.json').read_bytes())
    signatures={}
    for row in value.get('projections',[]):
        validate_public_projection(row)
        path=Path(directory)/(row['weightId']+'.i8')
        if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Regular owned Flash coefficient cache required')
        signatures[path]=file_signature(path)
    raw=validate_component(directory,CONFIG,GEOMETRY,semantics(value.get('clientGraphId')),versions(),source=TEXT)
    if any(file_signature(path)!=signature for path,signature in signatures.items()):raise ValueError('Flash coefficient cache changed during admission')
    return FlashTextPlan(raw)


def write_plan(directory,prepared):
    if list(prepared.order)!=list(GEOMETRY):raise ValueError('Complete installed Flash operation order required')
    prepared.require_graph()
    write_component(directory,CONFIG,prepared.definitions(),semantics(prepared.identity),versions(),source=TEXT)
    return load_plan(directory)
