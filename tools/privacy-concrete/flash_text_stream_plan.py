"""Explicit bounded Flash research storage; no production or security qualification.

Provider manifest parsing is deliberately separate from client admission. Client
admission additionally requires a complete in-memory byte-derived compiler proof.
Public tiles accompany ciphertext; neither endpoint needs a full disk cache.
"""
import json
import os
from pathlib import Path
import tempfile
from flash_text_ntt12_check_plan import CONFIG as CHECK_CONFIG,PROGRAM as CHECK_PROGRAM
from flash_text_plan import GEOMETRY
from flash_source_scopes import TEXT
from flash_component_plan import component_versions,validate_component_manifest,write_component
from full_plan import FullPlan,canonical
from protocol import digest,require_active,parse_unique_json
from public_weights import PreparedPublicTiles
from public_tile_stream import CONTRACT
from plan_backends import STREAMED_PROFILE

PROFILE=STREAMED_PROFILE
CONFIG=dict(CHECK_CONFIG,profile=PROFILE,schema='doppler.flash-text-stream-check-policy/v1',
    planSchema='doppler.flash-text-stream-check-plan/v1',protocol='concrete-glwe-flash-text-stream-linear-check/v1')
PROGRAM=dict(CHECK_PROGRAM,publicCoefficientStorage=CONTRACT)
PROGRAM_ID=digest(canonical(PROGRAM))
MAX_MANIFEST_BYTES=64*1024*1024


def versions():return component_versions(CONFIG)


def semantics(graph_id,digests):
    if (not isinstance(graph_id,str) or len(graph_id)!=64 or any(c not in '0123456789abcdef' for c in graph_id)
            or not isinstance(digests,dict) or not digests):raise ValueError('Exact graph and complete public tile identities required')
    for key,value in digests.items():
        if any(not isinstance(item,str) or len(item)!=64 or any(c not in '0123456789abcdef' for c in item) for item in (key,value)):
            raise ValueError('Canonical public tile digests required')
    per_step=sum(width*height*2 for width,height in GEOMETRY.values())
    return dict(programId=PROGRAM_ID,clientGraphId=graph_id,publicCoefficientStorage=CONTRACT,publicTileDigests=digests,
        publicCoefficientBytesPerStep=per_step,publicCoefficientBytesPerRoute=per_step*CONFIG['steps'],
        securityQualified=False,fullFlashModelQualified=False)


class FlashTextStreamPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_provider_plan(directory):
    path=Path(directory)/'plan.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_MANIFEST_BYTES:
        raise ValueError('Bounded regular streamed public manifest required')
    with path.open('rb') as handle:raw=handle.read(MAX_MANIFEST_BYTES+1)
    if len(raw)>MAX_MANIFEST_BYTES:raise ValueError('Public streamed manifest limit exceeded')
    value=parse_unique_json(raw)
    if not isinstance(value,dict):raise ValueError('Installed public streamed manifest required')
    meaning=semantics(value.get('clientGraphId'),value.get('publicTileDigests'))
    validate_component_manifest(raw,CONFIG,GEOMETRY,meaning,versions(),source=TEXT)
    plan=FlashTextStreamPlan(raw)
    if set(value['publicTileDigests'])!={tile['id'] for tile in plan.schedule()}:
        raise ValueError('Complete exact streamed tile route required')
    return plan


def load_client_plan(directory,prepared):
    # No flag or deserialized dictionary may replace the actual compiler record.
    if type(prepared) is not PreparedPublicTiles:raise ValueError('Client-owned complete byte-derived public tile binding required')
    plan=load_provider_plan(directory)
    if dict(prepared.digests_for(plan))!=plan.definition()['publicTileDigests']:
        raise ValueError('Streamed public manifest differs from verified client bytes')
    return plan


def write_plan(directory,prepared,*,cancelled=None):
    from flash_prepared_graph import PreparedFlashGraph,EPHEMERAL_STORAGE
    require_active(cancelled)
    if (type(prepared) is not PreparedFlashGraph or prepared.storage!=EPHEMERAL_STORAGE
            or prepared.quantization_bits!=12 or list(prepared.order)!=list(GEOMETRY)):
        raise ValueError('Complete installed evictable twelve-bit Flash preparation required')
    prepared.require_graph();root=Path(directory);path=root/'plan.json'
    with tempfile.TemporaryDirectory(dir=root,prefix='flash-public-manifest-') as temporary:
        write_component(temporary,CONFIG,prepared.definitions(),semantics(prepared.identity,prepared.public_tile_digests()),versions(),source=TEXT)
        plan=load_provider_plan(temporary);binding=prepared.bind_public_tiles(plan)
        require_active(cancelled)
        try:os.link(Path(temporary)/'plan.json',path)
        except FileExistsError:
            if (path.is_symlink() or not path.is_file() or path.stat().st_size!=len(plan.encoded)
                    or path.read_bytes()!=plan.encoded):raise ValueError('Existing public streamed manifest differs')
        require_active(cancelled)
    return load_client_plan(root,binding),binding
