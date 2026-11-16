"""Actual public signed message-domain boundary; not independent security qualification."""
import argparse
import json
import numpy as np
import concrete_ml_extensions as ext
from full_ntt12_plan import parameters
from profiles import validate_declared_arithmetic_capacity


def run():
    policy=parameters();spec=policy['parameters'];width=2048
    worst=validate_declared_arithmetic_capacity(width,12,spec['bits_reserved_for_computation'])
    definition=ext.MatmulCryptoParameters.deserialize(json.dumps(spec));secret,key=ext.cpu_create_private_key(definition);ext.cpu_prepare_ntt_key(key)
    inputs=np.full((1,width),4095,dtype=np.uint64);weights=np.tile(np.array([[-2048,2047]],dtype=np.int64),(width,1))
    expected=inputs.astype(np.int64)@weights
    encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=definition,data=inputs)
    result=ext.cpu_matrix_multiplication_ntt(encrypted,weights.astype(np.uint64),key)
    replay=ext.cpu_matrix_multiplication_ntt(encrypted,weights.astype(np.uint64),key)
    if result.serialize()!=replay.serialize():raise ValueError('12-bit domain result is not replay deterministic')
    actual=ext.decrypt_matrix(result,secret,definition,2).astype(np.int64)
    error=max(abs(int(left)-int(right)) for left,right in zip(actual.reshape(-1),expected.reshape(-1)))
    # A public numerical diagnostic criterion, not a security/noise-failure guarantee.
    if error>4096:raise ValueError('12-bit public boundary exceeds its declared diagnostic criterion')
    return dict(schema='doppler.public-qwen12-message-domain-probe/v1',productionBackendAvailable=False,securityQualified=False,fullModelExecuted=False,
        quantizationBits=12,messageBits=spec['bits_reserved_for_computation'],signedMessageLimit=2**(spec['bits_reserved_for_computation']-1),
        worstRawDot=worst,actualRawDots=actual.tolist(),expectedRawDots=expected.tolist(),maxIntegerError=error,diagnosticErrorCriterion=4096,
        deterministicReplay=True,evaluationKeyBytes=len(key.serialize()),interpretation='Disposable public boundary values; no model quality, privacy or malicious-security qualification')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir');parser.add_argument('--cache-dir');parser.parse_args();print(json.dumps(run(),indent=2))
