"""Explicit bounded Flash research storage; no production or security qualification.

Provider manifest parsing is deliberately separate from client admission. Client
admission additionally requires a complete in-memory byte-derived compiler proof.
Public tiles accompany ciphertext; neither endpoint needs a full disk cache.
"""
from flash_text_ntt12_check_plan import CONFIG as CHECK_CONFIG,PROGRAM as CHECK_PROGRAM
from flash_text_plan import GEOMETRY
from flash_source_scopes import TEXT
from flash_component_plan import component_versions
from full_plan import FullPlan,canonical
from protocol import digest
from coefficient_plan import public_semantics,read_provider_plan,read_client_plan,publish_plan,MAX_MANIFEST_BYTES
from public_tile_stream import CONTRACT
from plan_backends import STREAMED_PROFILE

PROFILE=STREAMED_PROFILE
CONFIG=dict(CHECK_CONFIG,profile=PROFILE,schema='doppler.flash-text-stream-check-policy/v2',
    planSchema='doppler.flash-text-stream-check-plan/v2',protocol='concrete-glwe-flash-text-stream-linear-check/v1')
PROGRAM=dict(CHECK_PROGRAM,publicCoefficientStorage=CONTRACT)
PROGRAM_ID=digest(canonical(PROGRAM))


def versions():return component_versions(CONFIG)


def semantics(graph_id,digests):
    return public_semantics(graph_id,digests,GEOMETRY,PROGRAM_ID,CONTRACT,CONFIG['steps'])


class FlashTextStreamPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_provider_plan(directory):
    return read_provider_plan(directory,CONFIG,GEOMETRY,TEXT,versions,semantics,FlashTextStreamPlan)


def load_client_plan(directory,prepared):
    return read_client_plan(directory,prepared,load_provider_plan)


def write_plan(directory,prepared,*,cancelled=None):
    return publish_plan(directory,prepared,CONFIG,GEOMETRY,TEXT,versions,semantics,
        load_provider_plan,load_client_plan,cancelled=cancelled)
