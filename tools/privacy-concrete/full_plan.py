"""Immutable, source-pinned all-linear Qwen execution plan with width-safe tiles."""
from dataclasses import dataclass
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from protocol import POLICY, binding, digest
from profiles import validate_arithmetic_capacity

FULL_POLICY = json.loads(Path(__file__).with_name('full-policy.json').read_text())
SCHEMA, PROTOCOL = FULL_POLICY['planSchema'], FULL_POLICY['protocol']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def ranges(width, limit):
    if type(width) is not int or width <= 0 or type(limit) is not int or limit <= 0:
        raise ValueError('Invalid explicit tile geometry')
    return tuple((start, min(start + limit, width)) for start in range(0, width, limit))


def tiles(projection, input_limit=2048, output_limit=2048):
    result = []
    for begin_out, end_out in ranges(projection['outputWidth'], output_limit):
        for begin_in, end_in in ranges(projection['inputWidth'], input_limit):
            validate_arithmetic_capacity(end_in - begin_in, 8)
            value = dict(name=projection['name'], weightId=projection['weightId'],
                         inputBegin=begin_in, inputEnd=end_in, outputBegin=begin_out, outputEnd=end_out)
            result.append(dict(value, id=digest(canonical(value))))
    return tuple(result)


@dataclass(frozen=True)
class FullPlan:
    encoded: bytes

    @property
    def protocol_id(self):return PROTOCOL

    @property
    def identity(self):
        return digest(self.encoded)

    def definition(self):
        return json.loads(self.encoded)

    def frame(self, session, key_id, sequence, kind, tile_id=''):
        value = binding(self.definition()['modelId'], session, key_id, sequence, kind, tile_id)
        value.update(schema=PROTOCOL, policyId=self.identity)
        return value

    def schedule(self):
        value = self.definition()
        return tuple(tile for projection in value['projections']
                     for tile in tiles(projection, value['inputTileWidth'], value['outputTileWidth']))


def load_plan(directory):
    directory = Path(directory)
    raw = (directory / 'plan.json').read_bytes()
    value = json.loads(raw)
    source = json.loads(Path(__file__).with_name('qwen05b.json').read_text())['source']
    if (raw != canonical(value) or value['schema'] != SCHEMA or value['protocol'] != PROTOCOL
            or value['source'] != source or value['productionBackendAvailable'] is not False
            or any(value[key] != FULL_POLICY[key] for key in ['steps','contextBucket','inputTileWidth','outputTileWidth','operationTimeoutSeconds','cpuThreads'])
            or len(value['projections']) != 169):
        raise ValueError('Unsupported full-model plan')
    if value['implementationVersions'] != implementation_versions():
        raise ValueError('Full-model implementation version mismatch')
    common = {key: POLICY[key] for key in ['dependencies', 'parameters', 'evaluationBackend']}
    if value['parameters'] != common:
        raise ValueError('Unqualified full-model parameters')
    expected_names = []
    for layer in range(24):
        expected_names += [f'model.layers.{layer}.self_attn.{name}_proj' for name in ['q','k','v','o']]
        expected_names += [f'model.layers.{layer}.mlp.{name}_proj' for name in ['gate','up','down']]
    expected_names.append('lm_head')
    if [projection['name'] for projection in value['projections']] != expected_names:
        raise ValueError('Incomplete or reordered full-model route')
    shapes={'q_proj':(896,896),'k_proj':(896,128),'v_proj':(896,128),'o_proj':(896,896),
            'gate_proj':(896,4864),'up_proj':(896,4864),'down_proj':(4864,896),'lm_head':(896,151936)}
    for projection in value['projections']:
        expected=shapes[projection['name'].rsplit('.',1)[-1]]
        if (projection['inputWidth'],projection['outputWidth']) != expected:
            raise ValueError('Unsupported full-model geometry')
        path = directory / (projection['weightId'] + '.i8')
        if len(projection['weightId']) != 64 or any(c not in '0123456789abcdef' for c in projection['weightId']):
            raise ValueError('Invalid public weight identity')
        if path.stat().st_size != projection['inputWidth'] * projection['outputWidth']:
            raise ValueError('Invalid public weight geometry')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != projection['weightId']:
                raise ValueError('Public model cache corruption')
        tiles(projection, value['inputTileWidth'], value['outputTileWidth'])
    model_id = digest(canonical(dict(source=source, projections=value['projections'])))
    if value['modelId'] != model_id:
        raise ValueError('Full-model identity mismatch')
    return FullPlan(raw)


def implementation_versions():
    actual = {}
    for dependency, expected in POLICY['dependencies'].items():
        found = version(dependency)
        allowed = FULL_POLICY['dependencyVariants'].get(dependency, [expected])
        if found not in allowed:
            raise ValueError('Pinned full-model dependencies required')
        actual[dependency] = found
    return actual
