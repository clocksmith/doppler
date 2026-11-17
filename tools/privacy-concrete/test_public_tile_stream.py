"""Pure byte proof and wire framing tests, no native model or ciphertext claims."""
from contextlib import contextmanager
import io
import struct
import threading
import unittest
from types import SimpleNamespace
from public_weights import PreparedPublicProjection,PreparedPublicTiles
from public_tile_stream import encode,decode
from full_plan import tiles
from protocol import digest
from weight_encoding import encoding_for_bits


class PublicTileStreamTests(unittest.TestCase):
    def fixture(self):
        raw=struct.pack('<12h',*range(-6,6));row=dict(name='projection',weightId=digest(raw),inputWidth=3,outputWidth=4)
        definition=dict(projections=[row],inputTileWidth=2,outputTileWidth=3,
            parameters=dict(weightEncoding=encoding_for_bits(12)['id'],quantizationBits=12))
        proof=PreparedPublicProjection(row,definition,io.BytesIO(raw))
        return raw,row,definition,proof

    def test_byte_derived_proofs_survive_eviction_and_bind_complete_plan(self):
        raw,row,definition,proof=self.fixture()
        plan=SimpleNamespace(definition=lambda:definition)
        binding=PreparedPublicTiles.from_projections(plan,[proof])
        @contextmanager
        def source(*args):yield io.BytesIO(raw)
        self.assertEqual(binding.digests_for(plan),PreparedPublicTiles(plan,source).digests_for(plan))
        for fake in ({'validated':True,'entries':proof.entries},[],[proof,proof]):
            with self.assertRaises((ValueError,TypeError)):PreparedPublicTiles.from_projections(plan,fake)
        changed=dict(definition,inputTileWidth=1)
        with self.assertRaises(ValueError):PreparedPublicTiles.from_projections(SimpleNamespace(definition=lambda:changed),[proof])
        changed=dict(definition,modelId='changed')
        with self.assertRaises(ValueError):binding.digests_for(SimpleNamespace(definition=lambda:changed))

    def test_wrong_whole_bytes_or_cancelled_proof_is_unavailable(self):
        raw,row,definition,_=self.fixture();event=threading.Event();event.set()
        for payload,cancel in ((bytes(len(raw)),None),(raw,event)):
            with self.assertRaises((ValueError,InterruptedError)):
                PreparedPublicProjection(row,definition,io.BytesIO(payload),cancelled=cancel)

    def test_versioned_tile_envelope_contains_exact_public_bytes_and_ciphertext(self):
        raw,row,definition,proof=self.fixture();encoding=encoding_for_bits(12)
        for tile in tiles(row,2,3):
            public=b''.join(raw[(i*4+tile['outputBegin'])*2:(i*4+tile['outputEnd'])*2] for i in range(tile['inputBegin'],tile['inputEnd']))
            expected=dict(proof.entries)[tile['id']];cipher=b'opaque-ciphertext'
            payload=encode(public,cipher,tile,encoding,expected)
            self.assertEqual(decode(payload,tile,encoding,expected),(public,cipher))
            for changed in (payload[1:],payload[:9],payload[:9]+bytes(len(public))+cipher,b'legacy'+payload):
                with self.assertRaises(ValueError):decode(changed,tile,encoding,expected)
            with self.assertRaises(ValueError):encode(public,b'',tile,encoding,expected)

if __name__=='__main__':unittest.main()
