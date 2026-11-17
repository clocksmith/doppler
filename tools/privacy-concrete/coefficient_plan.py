"""Shared admission/publication for explicitly installed coefficient storage backends.

Public manifest validation never substitutes for client-owned byte proofs.
Callers supply installed code/policy, never advertisement-selected callbacks.
"""
import os
from pathlib import Path
import tempfile
from flash_component_plan import validate_component_manifest,write_component
from protocol import parse_unique_json,require_active
from public_weights import PreparedPublicTiles

MAX_MANIFEST_BYTES=64*1024*1024


def public_semantics(graph_id,digests,geometry,program_id,contract,steps,*,wire_bytes=None):
    if (not isinstance(graph_id,str) or len(graph_id)!=64 or any(c not in '0123456789abcdef' for c in graph_id)
            or not isinstance(digests,dict) or not digests):raise ValueError('Exact graph and complete public tile identities required')
    for key,value in digests.items():
        if any(not isinstance(item,str) or len(item)!=64 or any(c not in '0123456789abcdef' for c in item) for item in (key,value)):
            raise ValueError('Canonical public tile digests required')
    per_step=sum(width*height*2 for width,height in geometry.values()) if wire_bytes is None else wire_bytes
    return dict(programId=program_id,clientGraphId=graph_id,publicCoefficientStorage=contract,publicTileDigests=digests,
        publicCoefficientBytesPerStep=per_step,publicCoefficientBytesPerRoute=per_step*steps,
        securityQualified=False,fullFlashModelQualified=False)


def read_provider_plan(directory,policy,geometry,source,versions,semantics,plan_type):
    path=Path(directory)/'plan.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_MANIFEST_BYTES:
        raise ValueError('Bounded regular public coefficient manifest required')
    with path.open('rb') as handle:raw=handle.read(MAX_MANIFEST_BYTES+1)
    if len(raw)>MAX_MANIFEST_BYTES:raise ValueError('Public coefficient manifest limit exceeded')
    value=parse_unique_json(raw)
    if not isinstance(value,dict):raise ValueError('Installed public coefficient manifest required')
    meaning=semantics(value.get('clientGraphId'),value.get('publicTileDigests'))
    validate_component_manifest(raw,policy,geometry,meaning,versions(),source=source)
    plan=plan_type(raw)
    if set(value['publicTileDigests'])!={tile['id'] for tile in plan.schedule()}:
        raise ValueError('Complete exact public coefficient tile route required')
    return plan


def read_client_plan(directory,prepared,provider_loader):
    if type(prepared) is not PreparedPublicTiles:raise ValueError('Client-owned complete byte-derived public tile binding required')
    plan=provider_loader(directory)
    if dict(prepared.digests_for(plan))!=plan.definition()['publicTileDigests']:
        raise ValueError('Public coefficient manifest differs from verified client bytes')
    return plan


def publish_plan(directory,prepared,policy,geometry,source,versions,semantics,provider_loader,client_loader,*,cancelled=None):
    from flash_prepared_graph import PreparedFlashGraph,EPHEMERAL_STORAGE
    require_active(cancelled)
    if (type(prepared) is not PreparedFlashGraph or prepared.storage!=EPHEMERAL_STORAGE
            or prepared.quantization_bits!=12 or list(prepared.order)!=list(geometry)):
        raise ValueError('Complete installed evictable twelve-bit Flash preparation required')
    prepared.require_graph();root=Path(directory);path=root/'plan.json'
    with tempfile.TemporaryDirectory(dir=root,prefix='flash-public-manifest-') as temporary:
        write_component(temporary,policy,prepared.definitions(),semantics(prepared.identity,prepared.public_tile_digests()),versions(),source=source)
        plan=provider_loader(temporary);binding=prepared.bind_public_tiles(plan)
        require_active(cancelled)
        try:os.link(Path(temporary)/'plan.json',path)
        except FileExistsError:
            if (path.is_symlink() or not path.is_file() or path.stat().st_size!=len(plan.encoded)
                    or path.read_bytes()!=plan.encoded):raise ValueError('Existing public coefficient manifest differs')
        require_active(cancelled)
    return client_loader(root,binding),binding
