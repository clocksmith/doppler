"""No silent parameter or payload expansion for existing research routes."""
import io
import unittest
from flash_component_plan import component_parameters
from protocol import POLICY,payload_limit,send,binding


class PrecisionAdmissionTests(unittest.TestCase):
    def test_precision_is_an_explicit_installed_profile_and_keeps_encryption_noise(self):
        legacy=component_parameters({})
        self.assertEqual(legacy['parameters'],POLICY['parameters']);self.assertNotIn('maxPayloadBytes',legacy)
        precise=component_parameters(dict(parameterProfile='glwe-precision-48-level4/v1'))
        for key in ['glwe_encryption_noise_distribution_stdev','pks_noise_distrubution_stdev','polynomial_size','bits_reserved_for_computation','ciphertext_modulus_bit_count']:
            self.assertEqual(precise['parameters'][key],legacy['parameters'][key])
        self.assertEqual(precise['parameters']['packing_ks_level'],4)
        self.assertEqual(precise['parameters']['input_storage_ciphertext_modulus'],48)
        for value in [None,'advertised-unreviewed-params']:
            with self.assertRaises(ValueError):component_parameters(dict(parameterProfile=value))

    def test_payload_limits_are_explicit_bounded_and_cannot_be_booleans(self):
        self.assertEqual(payload_limit(None),150000000);self.assertEqual(payload_limit(300000000),300000000)
        for value in [True,-1,0,300000001,'300000000']:
            with self.assertRaises(ValueError):payload_limit(value)
        output=io.BytesIO()
        with self.assertRaises(ValueError):send(output,binding('m','s','k',1,'evaluate'),b'three',max_payload_bytes=4)
        self.assertEqual(output.getvalue(),b'')


if __name__=='__main__':unittest.main()
