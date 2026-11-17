"""Compare bounded preparation with the observed whole-projection clear reference.

The installed pin is public build-time evidence, never provider metadata. This
gate runs before keys or ciphertexts and does not qualify encryption or integrity.
"""
import json
from pathlib import Path
from full_plan import canonical
from protocol import digest,require_active

REFERENCE=json.loads(Path(__file__).with_name('flash-checkpoint-reference.json').read_text())


def require_checkpoint_reference(prepared,*,cancelled=None):
    from flash_prepared_graph import PreparedFlashGraph
    from flash_source_scopes import TEXT
    from flash_text_plan import GEOMETRY
    require_active(cancelled)
    if type(prepared) is not PreparedFlashGraph:raise ValueError('Client-owned complete Flash preparation required')
    prepared.require_graph()
    if (REFERENCE['schema']!='doppler.flash-clear-coefficient-reference/v1'
            or any(REFERENCE[key] is not False for key in ('encryptedExecution','productionBackendAvailable','securityQualified'))
            or REFERENCE['sourceRevision']!=TEXT['revision'] or REFERENCE['sourceId']!=digest(canonical(TEXT))
            or prepared.identity!=prepared.graph.identity or prepared.quantization_bits!=REFERENCE['quantizationBits']
            or tuple(prepared.order)!=tuple(GEOMETRY) or len(prepared.order)!=REFERENCE['projections']):
        raise ValueError('Installed clear reference differs from the prepared Flash source or graph')
    definitions=prepared.definitions()
    if len(definitions)!=len(prepared.order):raise ValueError('Complete ordered reference projection definitions required')
    inventory=[]
    for name,row in zip(prepared.order,definitions):
        require_active(cancelled)
        if (row.get('name')!=name or (row.get('inputWidth'),row.get('outputWidth'))!=GEOMETRY[name]
                or not isinstance(row.get('weightId'),str) or len(row['weightId'])!=64
                or any(c not in '0123456789abcdef' for c in row['weightId'])):
            raise ValueError('Exact ordered reference projection geometry and coefficient identity required')
        inventory.append(dict(name=name,weightId=row['weightId']))
    observed=digest(canonical(inventory));require_active(cancelled)
    if observed!=REFERENCE['orderedCoefficientDigest']:
        raise ValueError('Bounded coefficients differ from the observed whole-projection clear reference')
    return dict(pinId=digest(canonical(REFERENCE)),receiptRevision=REFERENCE['receiptRevision'],
        receiptSha256=REFERENCE['receiptSha256'],orderedCoefficientDigest=observed,projections=len(inventory),
        referenceClientGraphId=REFERENCE['referenceClientGraphId'],admittedClientGraphId=prepared.identity)
