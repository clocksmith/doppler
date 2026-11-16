"""Installed, immutable evaluator profiles. Remote advertisements cannot select or modify them."""
from dataclasses import dataclass
import json
from pathlib import Path
from protocol import POLICY, POLICY_ID, binding, digest


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    protocol: str
    policy_id: str
    input_width: int
    output_width: int
    projection_count: int
    steps: int
    expected_weight_ids: tuple
    specification_json: str

    def frame(self, *args):
        result = binding(*args)
        result.update(schema=self.protocol, policyId=self.policy_id)
        return result

    def specification(self):
        return json.loads(self.specification_json)


def load_profile(name='tiny'):
    if name == 'tiny':
        return EvaluationProfile(name, POLICY['protocol'], POLICY_ID, POLICY['matrixWidth'],
                                 POLICY['matrixWidth'], POLICY['projectionCount'], POLICY['steps'], (), '{}')
    if name != 'qwen05b-k-projection':
        raise ValueError('Unsupported installed encrypted evaluation profile')
    raw = json.loads(Path(__file__).with_name('qwen05b.json').read_text())
    if (raw['id'] != name or raw['schema'] != 'doppler.encrypted-projection-profile/v1'
            or raw['productionBackendAvailable'] is not False or raw['projectionCount'] != 1):
        raise ValueError('Unqualified encrypted evaluation profile')
    validate_arithmetic_capacity(raw['projection']['inputWidth'], raw['quantizationBits'])
    canonical = json.dumps(raw, sort_keys=True, separators=(',', ':'))
    common = {key: POLICY[key] for key in ['dependencies', 'parameters', 'maxHeaderBytes', 'maxPayloadBytes', 'timeoutSeconds', 'evaluationBackend']}
    identity = digest(json.dumps(dict(protocolParameters=common, profile=raw), sort_keys=True, separators=(',', ':')).encode())
    return EvaluationProfile(name, raw['protocol'], identity, raw['projection']['inputWidth'],
                             raw['projection']['outputWidth'], raw['projectionCount'], raw['steps'],
                             (raw['projection']['quantizedWeightSha256'],), canonical)


def validate_arithmetic_capacity(input_width, quantization_bits):
    # Conservative signed raw-dot bound before local zero-point correction.
    if type(input_width) is not int or input_width <= 0 or quantization_bits != 8:
        raise ValueError('Unsupported encrypted arithmetic geometry')
    worst_case = input_width * (2 ** quantization_bits - 1) * 2 ** (quantization_bits - 1)
    signed_limit = 2 ** (POLICY['parameters']['bits_reserved_for_computation'] - 1)
    if worst_case >= signed_limit:
        raise ValueError('Unqualified encrypted integer range; reparameterization or a qualified tiled protocol required')
    return worst_case


def validate_declared_arithmetic_capacity(input_width,quantization_bits,computation_bits):
    """Explicit research domains; the legacy 8-bit validator remains unchanged."""
    if (type(input_width) is not int or input_width<=0 or type(quantization_bits) is not int
            or type(computation_bits) is not int or (quantization_bits,computation_bits) not in [(8,27),(12,35)]):
        raise ValueError('Unsupported declared encrypted arithmetic domain')
    worst=input_width*(2**quantization_bits-1)*2**(quantization_bits-1)
    if worst>=2**(computation_bits-1):raise ValueError('Declared encrypted raw dot can exceed its signed message domain')
    return worst
