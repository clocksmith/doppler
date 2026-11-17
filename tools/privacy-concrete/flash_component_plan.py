"""Shared immutable admission for source-pinned Flash research components."""
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
from full_plan import canonical,tiles,implementation_versions
from protocol import digest
from research_parameters import component_parameters
from weight_encoding import weight_encoding

WEIGHTS=json.loads(Path(__file__).with_name('flash-weights.json').read_text())
SOURCE_ID=digest(canonical(WEIGHTS))
LIMITS=('steps','contextBucket','inputTileWidth','outputTileWidth','cpuThreads','operationTimeoutSeconds')


def component_versions(policy):
    variant=policy.get('nativeVariant')
    if variant is None:found=implementation_versions()
    elif variant=='cpu-exact-ntt/v1':
        from native_runtime import ntt_versions
        found=ntt_versions()
    else:raise ValueError('Unsupported installed component arithmetic variant')
    for dependency,expected in policy['additionalDependencies'].items():
        found[dependency]=version(dependency)
        if found[dependency]!=expected:raise ValueError('Pinned Flash component dependencies required')
    return found


def validate_component(directory,policy,geometry,semantics,installed,source=WEIGHTS):
    directory=Path(directory);raw=(directory/'plan.json').read_bytes()
    value=validate_component_manifest(raw,policy,geometry,semantics,installed,source)
    encoding=weight_encoding(value['parameters'])
    for row in value['projections']:
        identity=row['weightId'];path=directory/(identity+encoding['suffix'])
        if path.stat().st_size!=row['inputWidth']*row['outputWidth']*encoding['bytes']:raise ValueError('Public component weight geometry changed')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=identity:raise ValueError('Public component weight corruption')
        if encoding['bytes']>1:
            from public_weights import mapped_public_matrix
            with mapped_public_matrix(path,(row['inputWidth'],row['outputWidth']),encoding['dtype']) as coefficients:
                bound=2**(encoding['quantizationBits']-1)
                if coefficients.min()< -bound or coefficients.max()>=bound:raise ValueError('Public coefficient exceeds its declared precision')
    return raw


def validate_component_manifest(raw,policy,geometry,semantics,installed,source=WEIGHTS):
    """Validate installed structure only; this never admits client public bytes."""
    source_id=digest(canonical(source));value=json.loads(raw)
    required=dict(schema=policy['planSchema'],protocol=policy['protocol'],sourceId=source_id,source=source,
                  productionBackendAvailable=False,implementationVersions=installed,
                  parameters=component_parameters(policy),
                  **{key:policy[key] for key in LIMITS},**semantics)
    if (raw!=canonical(value) or set(value)!=set(required)|{'projections','modelId'}
            or any(canonical(value[key])!=canonical(expected) for key,expected in required.items())
            or [row['name'] for row in value['projections']]!=list(geometry)):
        raise ValueError('Unsupported source-pinned Flash component plan')
    # Legacy components keep one signed byte per coefficient; a declared wider
    # encoding binds its own suffix, size, range and message-domain capacity.
    encoding=weight_encoding(required['parameters'])
    from public_weights import validate_public_projection
    for row in value['projections']:
        validate_public_projection(row)
        if (row['inputWidth'],row['outputWidth'])!=geometry[row['name']]:raise ValueError('Unsupported Flash component geometry')
        identity=row['weightId']
        if not isinstance(identity,str) or len(identity)!=64 or any(char not in '0123456789abcdef' for char in identity):
            raise ValueError('Public component weight identity required')
        if encoding['bytes']>1:
            from profiles import validate_declared_arithmetic_capacity
            from full_plan import ranges
            for begin,end in ranges(row['inputWidth'],value['inputTileWidth']):
                validate_declared_arithmetic_capacity(end-begin,encoding['quantizationBits'],required['parameters']['parameters']['bits_reserved_for_computation'])
        tiles(row,value['inputTileWidth'],value['outputTileWidth'])
    model_id=digest(canonical(dict(sourceId=source_id,**semantics,projections=value['projections'])))
    if value['modelId']!=model_id:raise ValueError('Public component model identity changed')
    return value


def write_component(directory,policy,definitions,semantics,installed,source=WEIGHTS):
    """One owner for canonical component identity; admission still revalidates stored public bytes."""
    source_id=digest(canonical(source))
    value=dict(schema=policy['planSchema'],protocol=policy['protocol'],sourceId=source_id,source=source,
               productionBackendAvailable=False,implementationVersions=installed,
               parameters=component_parameters(policy),
               **{key:policy[key] for key in LIMITS},**semantics,projections=definitions,
               modelId=digest(canonical(dict(sourceId=source_id,**semantics,projections=definitions))))
    (Path(directory)/'plan.json').write_bytes(canonical(value))
