"""Actual first full-attention Flash layer; explicit NTT route and client QSA/rotary operations."""
import json
from pathlib import Path
from full_plan import FullPlan,canonical
from flash_component_plan import validate_component,component_versions
from flash_source_scopes import source_spec
from flash_reference import SOURCE
from protocol import digest

CONFIG=json.loads(Path(__file__).with_name('flash-qsa-policy.json').read_text())
ORDER=tuple(CONFIG['projectionOrder']);WEIGHTS=source_spec('full-attention');SOURCE_ID=digest(canonical(WEIGHTS))
PREFIX='model.language_model.layers.3.self_attn.'
GEOMETRY={name:(WEIGHTS['tensors'][PREFIX+name+'.weight']['shape'][1],WEIGHTS['tensors'][PREFIX+name+'.weight']['shape'][0]) for name in ORDER}
PROGRAM=dict(schema='doppler.flash-first-qsa-client-program/v1',operatorReferenceId=digest(canonical(SOURCE)),sourceConfigId=WEIGHTS['configSha256'],
             projectionOrder=list(ORDER),layerIndex=3,contextLimit=8,
             clientOperations=['text-default-rotary','eager-qsa-indexing','attention-softmax','normalization','sigmoid-output-gate'],
             requestState='tenant-privacy-model-key-request-mode-bound/v1')
PROGRAM_ID=digest(canonical(PROGRAM))


def versions():return component_versions(CONFIG)


class FlashQSAPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    return FlashQSAPlan(validate_component(directory,CONFIG,GEOMETRY,dict(programId=PROGRAM_ID),versions(),source=WEIGHTS))
