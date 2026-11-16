"""Exact fixed-public-expert component admission. It cannot identify or admit the full Flash model."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from full_plan import FullPlan, canonical, implementation_versions, tiles
from protocol import POLICY, digest

POLICY_EXPERT=json.loads(Path(__file__).with_name('flash-expert-policy.json').read_text())
WEIGHTS=json.loads(Path(__file__).with_name('flash-weights.json').read_text())
SOURCE_ID=digest(canonical(WEIGHTS))
GEOMETRY={'expert/0/gate_up':(2560,1280),'expert/0/down':(640,2560)}


class FlashExpertPlan(FullPlan):
    @property
    def protocol_id(self):return POLICY_EXPERT['protocol']


def versions():
    found=implementation_versions()
    for dependency,expected in POLICY_EXPERT['additionalDependencies'].items():
        found[dependency]=version(dependency)
        if found[dependency]!=expected:raise ValueError('Pinned Flash component dependencies required')
    return found


def load_plan(directory):
    directory=Path(directory);raw=(directory/'plan.json').read_bytes();value=json.loads(raw)
    if (raw!=canonical(value) or value['schema']!=POLICY_EXPERT['planSchema'] or value['protocol']!=POLICY_EXPERT['protocol']
            or value['sourceId']!=SOURCE_ID or canonical(value['source'])!=canonical(WEIGHTS) or value['productionBackendAvailable'] is not False
            or value['implementationVersions']!=versions()
            or any(type(value[key]) is not type(POLICY_EXPERT[key]) or value[key]!=POLICY_EXPERT[key] for key in ['expertIndex','steps','contextBucket','inputTileWidth','outputTileWidth','cpuThreads','operationTimeoutSeconds'])
            or canonical(value['parameters'])!=canonical({key:POLICY[key] for key in ['dependencies','parameters','evaluationBackend']})
            or [row['name'] for row in value['projections']]!=list(GEOMETRY)):
        raise ValueError('Unsupported fixed Flash expert component plan')
    for row in value['projections']:
        if (row['inputWidth'],row['outputWidth'])!=GEOMETRY[row['name']]:raise ValueError('Unsupported Flash expert geometry')
        identity=row['weightId']
        if not isinstance(identity,str) or len(identity)!=64 or any(char not in '0123456789abcdef' for char in identity):
            raise ValueError('Public component weight identity required')
        path=directory/(identity+'.i8')
        if path.stat().st_size!=row['inputWidth']*row['outputWidth']:raise ValueError('Public component weight geometry changed')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=identity:raise ValueError('Public component weight corruption')
        tiles(row,value['inputTileWidth'],value['outputTileWidth'])
    model_id=digest(canonical(dict(sourceId=SOURCE_ID,expertIndex=value['expertIndex'],projections=value['projections'])))
    if value['modelId']!=model_id:raise ValueError('Public component model identity changed')
    return FlashExpertPlan(raw)
