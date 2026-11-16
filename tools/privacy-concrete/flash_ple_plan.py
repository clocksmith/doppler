"""Installed actual PLE route; complete local table identity is part of the immutable model."""
import json
from pathlib import Path
from full_plan import FullPlan,canonical
from flash_component_plan import validate_component,component_versions
from flash_source_scopes import source_spec
from flash_reference import SOURCE
from protocol import digest

CONFIG=json.loads(Path(__file__).with_name('flash-ple-policy.json').read_text())
TABLE=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text());TABLE_ID=digest(canonical(TABLE))
WEIGHTS=source_spec('ple');SOURCE_ID=digest(canonical(WEIGHTS));PREFIX='model.language_model.layers.1.ple.'
ORDER=tuple(CONFIG['projectionOrder'])
GEOMETRY={name:(WEIGHTS['tensors'][PREFIX+name+'.weight']['shape'][1],WEIGHTS['tensors'][PREFIX+name+'.weight']['shape'][0]) for name in ORDER}
PROGRAM=dict(schema='doppler.flash-first-ple-client-program/v1',operatorReferenceId=digest(canonical(SOURCE)),sourceConfigId=WEIGHTS['configSha256'],
             localTableId=TABLE_ID,projectionOrder=list(ORDER),layerIndex=1,contextLimit=8,maxLookupBytes=CONFIG['maxLookupBytes'],
             clientOperations=['local-bf16-table-lookup','eos-aware-ngram-history','normalization','gating','dilated-depthwise-convolution'],
             requestState='tenant-privacy-model-key-request-mode-bound/v1')
PROGRAM_ID=digest(canonical(PROGRAM));SEMANTICS=dict(programId=PROGRAM_ID,localTableId=TABLE_ID)


def versions():return component_versions(CONFIG)


class FlashPLEPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    return FlashPLEPlan(validate_component(directory,CONFIG,GEOMETRY,SEMANTICS,versions(),source=WEIGHTS))
