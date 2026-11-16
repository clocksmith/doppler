"""Versioned public coefficient representations and explicitly declared integer domains."""
import struct
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from weight_encoding import weight_encoding,encoding_for_bits
from profiles import validate_arithmetic_capacity,validate_declared_arithmetic_capacity
from public_weights import bind_public_tiles
from full_plan import tiles
from protocol import digest
from plan_backends import qwen_backend


class WeightEncodingTests(unittest.TestCase):
    def test_legacy_encoding_and_unknown_or_inconsistent_encodings_fail_closed(self):
        self.assertEqual(weight_encoding({})['id'],'signed-int8')
        for parameters in [None,{'weightEncoding':None},{'weightEncoding':'advertised'},{'weightEncoding':'signed-int8','quantizationBits':12}]:
            with self.assertRaises(ValueError):weight_encoding(parameters)
        for bits in [True,16,None]:
            with self.assertRaises(ValueError):encoding_for_bits(bits)

    def test_wide_source_domain_does_not_expand_the_legacy_eight_bit_profile(self):
        self.assertEqual(validate_declared_arithmetic_capacity(2048,12,35),17175674880)
        with self.assertRaises(ValueError):validate_arithmetic_capacity(2048,12)
        with self.assertRaises(ValueError):validate_declared_arithmetic_capacity(2048,12,27)
        with self.assertRaises(ValueError):validate_declared_arithmetic_capacity(8192,12,35)
        policy,_,_,parameters=qwen_backend('qwen-all-linear-ntt12')
        self.assertEqual(policy['quantizationBits'],12);self.assertEqual(parameters['parameters']['bits_reserved_for_computation'],35)
        self.assertEqual(weight_encoding(parameters)['bytes'],2)
        legacy,_,_,original=qwen_backend('qwen-all-linear-ntt')
        self.assertEqual(legacy['quantizationBits'],8);self.assertEqual(original['parameters']['bits_reserved_for_computation'],27)
        self.assertNotEqual(policy['protocol'],legacy['protocol'])

    def test_int16_snapshots_bind_exact_byte_strides_without_int8_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            values=list(range(-1200,-1176));data=struct.pack('<24h',*values);identity=digest(data)
            path=Path(directory)/(identity+'.i16');path.write_bytes(data)
            row=dict(name='projection',weightId=identity,inputWidth=4,outputWidth=6)
            definition=dict(projections=[row],inputTileWidth=2,outputTileWidth=3,parameters=dict(weightEncoding='signed-int12-in-int16',quantizationBits=12))
            plan=SimpleNamespace(definition=lambda:definition);binding=bind_public_tiles(directory,plan);schedule=tiles(row,2,3)
            expected=struct.pack('<6h',*(values[:3]+values[6:9]))
            self.assertEqual(binding[schedule[0]['id']],digest(expected))
            path.write_bytes(bytes(len(data)))
            self.assertEqual(binding[schedule[0]['id']],digest(expected))
            with self.assertRaises(ValueError):bind_public_tiles(directory,plan)


if __name__=='__main__':unittest.main()
