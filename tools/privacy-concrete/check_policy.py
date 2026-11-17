"""Exact checked-integrity policy identities and immutable attempt snapshots.

No advertisement registers a policy. Source configuration changes require a new
compiled plan; a live attempt always retains its original immutable snapshot.
"""
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

POLICY=json.loads(Path(__file__).with_name('linear-check-policy.json').read_text())


def validate_policy(policy):
    fields={'schema','id','securityQualified','independentCryptographicReview','productionBackendAvailable',
        'checkVectors','weightBits','falseRejectLog2','replaySample','rejection','rejectedResponseSubstitute','noise','vectorScope'}
    sample=policy.get('replaySample',{}) if isinstance(policy,Mapping) else {}
    if (not isinstance(policy,Mapping) or set(policy)!=fields or not isinstance(sample,Mapping) or set(sample)!={'numerator','denominator'}
            or policy.get('schema')!='doppler.linear-check-policy/v1' or policy.get('id')!='secret-linear-check-sampled-replay/v1'
            or any(policy.get(name) is not False for name in ('securityQualified','independentCryptographicReview','productionBackendAvailable'))
            or type(policy.get('checkVectors')) is not int or not 16<=policy['checkVectors']<=64
            or policy.get('weightBits')!=16 or type(policy.get('falseRejectLog2')) is not int or not 20<=policy['falseRejectLog2']<=80
            or type(sample.get('numerator')) is not int or type(sample.get('denominator')) is not int
            or not 1<=sample['numerator']<=sample['denominator']<=64 or policy.get('vectorScope')!='tile-step-row/v1'
            or policy.get('rejection')!='deferred-until-route-close/v1' or policy.get('rejectedResponseSubstitute')!='zero'):
        raise ValueError('Unsupported installed linear-check policy')
    noise=policy.get('noise')
    if not isinstance(noise,Mapping) or not noise:raise ValueError('Measured honest-noise models required')
    for profile,value in noise.items():
        if (not isinstance(profile,str) or not 0<len(profile)<=256 or not isinstance(value,Mapping)
                or set(value)!={'varianceBase','variancePerOutput','margin','evidence'}
                or any(type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) for key in ('varianceBase','variancePerOutput','margin'))
                or value['varianceBase']<0 or value['variancePerOutput']<=0 or not 1<=value['margin']<=4
                or not isinstance(value.get('evidence'),str) or not 0<len(value['evidence'])<=512):
            raise ValueError('No measured honest-noise model is installed for this parameter profile')
    return policy


def _plain(value):
    if isinstance(value,Mapping):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(item) for item in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('JSON check-policy metadata required')


def _encoded(policy):
    value=_plain(policy);validate_policy(value)
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def policy_identity(policy):return hashlib.sha256(_encoded(policy)).hexdigest()


def _freeze(value):
    if isinstance(value,dict):return MappingProxyType({key:_freeze(item) for key,item in value.items()})
    if isinstance(value,list):return tuple(_freeze(item) for item in value)
    return value


def freeze_policy(policy):
    encoded=_encoded(policy)
    return _freeze(json.loads(encoded)),hashlib.sha256(encoded).hexdigest()


POLICY_ID=policy_identity(POLICY)


def policy_binding(policy):
    return dict(integrity=validate_policy(policy)['id'],integrityPolicyId=policy_identity(policy))


def installed_binding():
    value=policy_binding(POLICY)
    if value['integrityPolicyId']!=POLICY_ID:raise ValueError('Installed check policy changed; a new compiled policy and plan are required')
    return value


def require_bound_policy(plan,parameter_profile):
    binding=installed_binding();snapshot,identity=freeze_policy(POLICY)
    if (identity!=binding['integrityPolicyId'] or parameter_profile not in snapshot['noise']
            or plan.definition().get('parameters',{}).get('integrityPolicyId')!=identity):
        raise ValueError('Exact installed check-policy digest must be bound into the admitted plan')
    return snapshot
