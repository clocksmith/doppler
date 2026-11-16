"""Opt-in actual GLWE width-overflow regression; whole-input quantization, no unsafe wide encrypted dot."""
import os
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE')=='1','opt-in pinned native crypto')
class WideNativeTests(unittest.TestCase):
    def test_native_tiles_reconstruct_a_dot_exceeding_the_old_signed_range(self):
        import numpy as np
        import concrete_ml_extensions as ext
        from full_plan import ranges
        parameters=ext.MatmulCryptoParameters.deserialize(ext.default_params())
        secret,key=ext.cpu_create_private_key(parameters)
        inputs=np.full((1,4864),255,dtype=np.uint64)
        weights=np.full((4864,2),-128,dtype=np.int64)
        reconstructed=np.zeros((1,2),dtype=np.int64)
        for begin,end in ranges(4864,2048):
            encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=np.ascontiguousarray(inputs[:,begin:end]))
            result=ext.cpu_matrix_multiplication(encrypted,np.ascontiguousarray(weights[begin:end]).astype(np.uint64),key)
            replay=ext.cpu_matrix_multiplication(encrypted,np.ascontiguousarray(weights[begin:end]).astype(np.uint64),key)
            self.assertEqual(result.serialize(),replay.serialize())
            reconstructed+=ext.decrypt_matrix(result,secret,parameters,2).astype(np.int64)
        expected=inputs.astype(np.int64)@weights
        self.assertLess(int(expected.min()),-(2**26))
        self.assertLessEqual(int(np.abs(reconstructed-expected).max()),128)


if __name__=='__main__':unittest.main()
