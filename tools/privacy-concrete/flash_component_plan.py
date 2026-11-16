"""Shared immutable admission for source-pinned Flash research components."""
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
from full_plan import canonical,tiles,implementation_versions
from protocol import POLICY,digest

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


def component_parameters(policy):
    variant=policy.get('nativeVariant')
    if variant is None:result=json.loads(canonical({key:POLICY[key] for key in ['dependencies','parameters','evaluationBackend']}))
    elif variant=='cpu-exact-ntt/v1':
        from native_runtime import ntt_parameters
        result=ntt_parameters()
    else:raise ValueError('Unsupported installed component arithmetic variant')
    if 'parameterProfile' in policy:
        profile_id=policy['parameterProfile']
        profile=json.loads(Path(__file__).with_name('precision-policy.json').read_text())
        if profile_id!=profile['id'] or profile['securityQualified'] is not False:raise ValueError('Unsupported research precision profile')
        result=dict(result,parameters=profile['parameters'],maxPayloadBytes=profile['maxPayloadBytes'])
    return result


def validate_component(directory,policy,geometry,semantics,installed,source=WEIGHTS):
    source_id=digest(canonical(source))
    directory=Path(directory);raw=(directory/'plan.json').read_bytes();value=json.loads(raw)
    required=dict(schema=policy['planSchema'],protocol=policy['protocol'],sourceId=source_id,source=source,
                  productionBackendAvailable=False,implementationVersions=installed,
                  parameters=component_parameters(policy),
                  **{key:policy[key] for key in LIMITS},**semantics)
    if (raw!=canonical(value) or set(value)!=set(required)|{'projections','modelId'}
            or any(canonical(value[key])!=canonical(expected) for key,expected in required.items())
            or [row['name'] for row in value['projections']]!=list(geometry)):
        raise ValueError('Unsupported source-pinned Flash component plan')
    for row in value['projections']:
        if (row['inputWidth'],row['outputWidth'])!=geometry[row['name']]:raise ValueError('Unsupported Flash component geometry')
        identity=row['weightId']
        if not isinstance(identity,str) or len(identity)!=64 or any(char not in '0123456789abcdef' for char in identity):
            raise ValueError('Public component weight identity required')
        path=directory/(identity+'.i8')
        if path.stat().st_size!=row['inputWidth']*row['outputWidth']:raise ValueError('Public component weight geometry changed')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=identity:raise ValueError('Public component weight corruption')
        tiles(row,value['inputTileWidth'],value['outputTileWidth'])
    model_id=digest(canonical(dict(sourceId=source_id,**semantics,projections=value['projections'])))
    if value['modelId']!=model_id:raise ValueError('Public component model identity changed')
    return raw


def write_component(directory,policy,definitions,semantics,installed,source=WEIGHTS):
    """One owner for canonical component identity; admission still revalidates stored public bytes."""
    source_id=digest(canonical(source))
    value=dict(schema=policy['planSchema'],protocol=policy['protocol'],sourceId=source_id,source=source,
               productionBackendAvailable=False,implementationVersions=installed,
               parameters=component_parameters(policy),
               **{key:policy[key] for key in LIMITS},**semantics,projections=definitions,
               modelId=digest(canonical(dict(sourceId=source_id,**semantics,projections=definitions))))
    (Path(directory)/'plan.json').write_bytes(canonical(value))
