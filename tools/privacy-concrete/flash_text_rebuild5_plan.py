"""Explicit five-row prefill for the public capital fixture; no prompt truncation.

Distinct immutable policy, program, plan and backend protocol. The one-row
profile remains unchanged; neither profile implies full-model qualification.
"""
from flash_text_rebuild_plan import CONFIG as BASE_CONFIG,PROGRAM as BASE_PROGRAM,source_semantics
from flash_text_plan import GEOMETRY
from flash_source_scopes import TEXT
from flash_component_plan import component_versions
from full_plan import FullPlan,canonical
from protocol import digest
from coefficient_plan import read_provider_plan,read_client_plan,publish_plan,MAX_MANIFEST_BYTES
from flash_provider_source import linear_source_spec
from plan_backends import REBUILD_PREFILL5_PROFILE

PROFILE=REBUILD_PREFILL5_PROFILE
CONFIG=dict(BASE_CONFIG,profile=PROFILE,contextBucket=5,schema='doppler.flash-text-rebuild-prefill5-check-policy/v1',
    planSchema='doppler.flash-text-rebuild-prefill5-check-plan/v1',protocol='concrete-glwe-flash-text-rebuild-prefill5-linear-check/v1')
PROGRAM=dict(BASE_PROGRAM,prefillRows=5)
PROGRAM_ID=digest(canonical(PROGRAM))


def versions():return component_versions(CONFIG)


def semantics(graph_id,digests):
    return source_semantics(graph_id,digests,GEOMETRY,PROGRAM_ID,CONFIG['steps'],linear_source_spec())


class FlashTextRebuild5Plan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_provider_plan(directory):
    return read_provider_plan(directory,CONFIG,GEOMETRY,TEXT,versions,semantics,FlashTextRebuild5Plan)


def load_client_plan(directory,prepared):
    return read_client_plan(directory,prepared,load_provider_plan)


def write_plan(directory,prepared,*,cancelled=None):
    return publish_plan(directory,prepared,CONFIG,GEOMETRY,TEXT,versions,semantics,
        load_provider_plan,load_client_plan,cancelled=cancelled)
