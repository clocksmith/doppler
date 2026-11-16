"""Opt-in exact-integer check for a separately declared, unqualified precision profile."""
import json
import os
import unittest
from flash_component_plan import component_parameters


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE')=='1','opt-in pinned native precision check')
class HighPrecisionTests(unittest.TestCase):
    def test_wide_signed_products_match_exact_reference_and_preserve_replay_bytes(self):
        import numpy as np
        import concrete_ml_extensions as ext
        profile=component_parameters(dict(parameterProfile='glwe-precision-48-level4/v1'))
        parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(profile['parameters']))
        secret,key=ext.cpu_create_private_key(parameters)
        self.assertGreater(len(key.serialize()),150000000)
        self.assertLess(len(key.serialize()),profile['maxPayloadBytes'])
        generator=np.random.default_rng(19)
        for _case in range(3):
            inputs=generator.integers(0,256,size=(1,2048),dtype=np.uint64)
            weights=generator.integers(-128,128,size=(2048,2),dtype=np.int64)
            encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=inputs)
            actual=ext.cpu_matrix_multiplication(encrypted,weights.astype(np.uint64),key)
            replay=ext.cpu_matrix_multiplication(encrypted,weights.astype(np.uint64),key)
            self.assertEqual(actual.serialize(),replay.serialize())
            decoded=ext.decrypt_matrix(actual,secret,parameters,2).astype(np.int64)
            np.testing.assert_array_equal(decoded,inputs.astype(np.int64)@weights)


if __name__=='__main__':unittest.main()
