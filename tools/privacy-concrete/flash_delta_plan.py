"""Actual first-layer Flash DeltaNet research route, independent of MoE and full-model profiles."""
from full_plan import FullPlan,canonical
from flash_component_plan import validate_component,component_versions
from flash_source_scopes import source_spec
from flash_reference import SOURCE
from flash_delta_program import POLICY_DELTA as CONFIG,ORDER
from protocol import digest

WEIGHTS=source_spec('linear-attention');SOURCE_ID=digest(canonical(WEIGHTS))
PREFIX='model.language_model.layers.0.linear_attn.'
GEOMETRY={name:(WEIGHTS['tensors'][PREFIX+name+'.weight']['shape'][1],WEIGHTS['tensors'][PREFIX+name+'.weight']['shape'][0]) for name in ORDER}
PROGRAM=dict(schema='doppler.flash-first-delta-client-program/v1',operatorReferenceId=digest(canonical(SOURCE)),
             sourceConfigId=WEIGHTS['configSha256'],projectionOrder=list(ORDER),
             clientOperations=['depthwise-causal-convolution','gated-delta-recurrence','normalization','activation'],
             requestState='tenant-privacy-model-key-request-mode-bound/v1')
PROGRAM_ID=digest(canonical(PROGRAM))


def versions():return component_versions(CONFIG)


class FlashDeltaPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    return FlashDeltaPlan(validate_component(directory,CONFIG,GEOMETRY,dict(programId=PROGRAM_ID),versions(),source=WEIGHTS))
