"""Full-text Flash research route on the twelve-bit NTT parameters with checked integrity.

Same installed geometry, source and client graph contract as the replay
profile in flash_text_plan. It differs in arithmetic (exact-NTT native variant,
twelve-bit coefficients in int16, 35-bit message domain) and in client
integrity (INTEGRITY.md). No full-model run, numerical or security
qualification exists for it; production admission stays closed.
"""
import stat
import json
from pathlib import Path
from full_plan import FullPlan,canonical
from flash_source_scopes import TEXT
from flash_text_plan import GEOMETRY,PROGRAM as REPLAY_PROGRAM
from flash_component_plan import component_versions,validate_component,write_component
from linear_check import POLICY as CHECK_POLICY
from protocol import digest
from public_weights import validate_public_projection
from research_parameters import component_parameters
from source_snapshot import file_signature
from weight_encoding import weight_encoding

CONFIG=dict(schema='doppler.flash-text-ntt12-check-policy/v1',profile='flash-text-all-linear-ntt12-check',
    planSchema='doppler.flash-text-ntt12-check-plan/v1',protocol='concrete-glwe-flash-text-ntt12-linear-check/v1',
    productionBackendAvailable=False,steps=4,contextBucket=1,inputTileWidth=2048,outputTileWidth=2048,
    cpuThreads=4,operationTimeoutSeconds=900,additionalDependencies={'safetensors':'0.8.0'},
    nativeVariant='cpu-exact-ntt/v1',parameterProfile='glwe-quant12-message35-storage48/v1',quantizationBits=12,
    integrity=CHECK_POLICY['id'])
PROGRAM=dict(REPLAY_PROGRAM,schema='doppler.flash-text-client-program/v1',integrity=CHECK_POLICY['id'],
    arithmetic='cpu-exact-ntt/v1+glwe-quant12-message35-storage48/v1')
PROGRAM_ID=digest(canonical(PROGRAM))


def semantics(graph_id):
    if not isinstance(graph_id,str) or len(graph_id)!=64 or any(c not in '0123456789abcdef' for c in graph_id):raise ValueError('Explicit client graph identity required')
    return dict(programId=PROGRAM_ID,clientGraphId=graph_id,securityQualified=False,fullFlashModelQualified=False)


def versions():return component_versions(CONFIG)


def parameters():return component_parameters(CONFIG)


class FlashTextNtt12CheckPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    value=json.loads((Path(directory)/'plan.json').read_bytes())
    suffix=weight_encoding(parameters())['suffix'];signatures={}
    for row in value.get('projections',[]):
        validate_public_projection(row)
        path=Path(directory)/(row['weightId']+suffix)
        if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Regular owned Flash coefficient cache required')
        signatures[path]=file_signature(path)
    raw=validate_component(directory,CONFIG,GEOMETRY,semantics(value.get('clientGraphId')),versions(),source=TEXT)
    if any(file_signature(path)!=signature for path,signature in signatures.items()):raise ValueError('Flash coefficient cache changed during admission')
    return FlashTextNtt12CheckPlan(raw)


def write_plan(directory,prepared):
    if list(prepared.order)!=list(GEOMETRY):raise ValueError('Complete installed Flash operation order required')
    prepared.require_graph()
    write_component(directory,CONFIG,prepared.definitions(),semantics(prepared.identity),versions(),source=TEXT)
    return load_plan(directory)
