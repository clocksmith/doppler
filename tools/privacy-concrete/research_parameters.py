"""One owner for installed research arithmetic parameters; no advertisement registration."""
import json
from pathlib import Path
from full_plan import canonical
from protocol import POLICY


def component_parameters(policy):
    variant=policy.get('nativeVariant')
    if variant is None:result=json.loads(canonical({key:POLICY[key] for key in ['dependencies','parameters','evaluationBackend']}))
    elif variant=='cpu-exact-ntt/v1':
        from native_runtime import ntt_parameters
        result=ntt_parameters()
    else:raise ValueError('Unsupported installed component arithmetic variant')
    if 'parameterProfile' in policy:
        profile_id=policy['parameterProfile']
        registry=json.loads(Path(__file__).with_name('precision-profiles.json').read_text())['profiles']
        if not isinstance(profile_id,str) or profile_id not in registry:raise ValueError('Unsupported installed research precision profile')
        profile=json.loads(Path(__file__).with_name(registry[profile_id]).read_text())
        if profile_id!=profile['id'] or profile['securityQualified'] is not False:raise ValueError('Unsupported research precision profile')
        result=dict(result,parameters=profile['parameters'],maxPayloadBytes=profile['maxPayloadBytes'])
        for key in ['quantizationBits','weightEncoding']:
            if key in profile:result[key]=profile[key]
    return result

