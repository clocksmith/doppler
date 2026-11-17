"""Zero coefficient transfer using an already provisioned, whole-verified checkpoint.

Distinct research profile, never a fallback from the streaming route. Client
byte proofs and checked integrity remain mandatory; no production qualification.
"""
from flash_text_ntt12_check_plan import CONFIG as CHECK_CONFIG,PROGRAM as CHECK_PROGRAM
from flash_text_plan import GEOMETRY
from flash_source_scopes import TEXT
from flash_component_plan import component_versions
from full_plan import FullPlan,canonical
from protocol import digest
from coefficient_plan import public_semantics,read_provider_plan,read_client_plan,publish_plan,MAX_MANIFEST_BYTES
from public_tile_stream import REBUILD_CONTRACT
from flash_provider_source import linear_source_spec
from plan_backends import REBUILD_PROFILE

PROFILE=REBUILD_PROFILE
CONFIG=dict(CHECK_CONFIG,profile=PROFILE,schema='doppler.flash-text-rebuild-check-policy/v2',
    planSchema='doppler.flash-text-rebuild-check-plan/v2',protocol='concrete-glwe-flash-text-rebuild-linear-check/v1')
PROGRAM=dict(CHECK_PROGRAM,publicCoefficientStorage=REBUILD_CONTRACT)
PROGRAM_ID=digest(canonical(PROGRAM))
CACHE_BUDGET_BYTES=2*1024**3
SOURCE_VERIFICATION_TIMEOUT_SECONDS=7200


def versions():return component_versions(CONFIG)


def source_semantics(graph_id,digests,geometry,program_id,steps,source):
    return dict(public_semantics(graph_id,digests,geometry,program_id,REBUILD_CONTRACT,steps,wire_bytes=0),
        logicalCoefficientBytesPerStep=sum(width*height*2 for width,height in geometry.values()),
        providerSourceId=digest(canonical(source)),providerSourceShardBytes=source['wholeShardBytes'],
        providerCoefficientCacheBudgetBytes=CACHE_BUDGET_BYTES,sourceVerificationTimeoutSeconds=SOURCE_VERIFICATION_TIMEOUT_SECONDS)


def semantics(graph_id,digests):
    return source_semantics(graph_id,digests,GEOMETRY,PROGRAM_ID,CONFIG['steps'],linear_source_spec())


class FlashTextRebuildPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_provider_plan(directory):
    return read_provider_plan(directory,CONFIG,GEOMETRY,TEXT,versions,semantics,FlashTextRebuildPlan)


def load_client_plan(directory,prepared):
    return read_client_plan(directory,prepared,load_provider_plan)


def write_plan(directory,prepared,*,cancelled=None):
    return publish_plan(directory,prepared,CONFIG,GEOMETRY,TEXT,versions,semantics,
        load_provider_plan,load_client_plan,cancelled=cancelled)
