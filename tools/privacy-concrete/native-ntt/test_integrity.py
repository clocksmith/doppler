"""Real native ciphertext rejection boundary, not model/transport/security qualification."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import concrete_ml_extensions as ext
import full_bridge
from full_admission import FullRouteAdmission
from full_plan import FullPlan,canonical
from full_status import ExecutionRejected
from native_runtime import ntt_source_id
from protocol import digest


class NativeNttIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ext.is_cuda_enabled() or ext.__doppler_ntt_build_id__!=ntt_source_id():raise ValueError('Exact installed CPU NTT source required')
        cls.parameters=ext.MatmulCryptoParameters.deserialize(ext.default_params())
        cls.secret,cls.key=ext.cpu_create_private_key(cls.parameters);ext.cpu_prepare_ntt_key(cls.key)
        cls.weights=np.arange(-16,16,dtype=np.int64).reshape(16,2).astype(np.uint64)
        cls.cipher=ext.encrypt_matrix(pkey=cls.secret,crypto_params=cls.parameters,data=np.arange(1,17,dtype=np.uint64).reshape(1,16))
        cls.good=ext.cpu_matrix_multiplication_ntt(cls.cipher,cls.weights,cls.key).serialize()
        cls.wrong=ext.cpu_matrix_multiplication_ntt(cls.cipher,np.zeros_like(cls.weights),cls.key).serialize()
        if cls.good==cls.wrong:raise ValueError('Public negative ciphertext fixture did not differ')
        cls.native_result_type=ext.CompressedResultEncryptedMatrix

    def bridge(self,response,event=None,cancel_during_replay=False):
        row=dict(name='public-test',weightId=digest(self.weights.view(np.int64).astype(np.int8).tobytes()),inputWidth=16,outputWidth=2)
        plan=FullPlan(canonical(dict(modelId='a'*64,steps=1,contextBucket=1,inputTileWidth=2048,outputTileWidth=2048,projections=[row])))
        bridge=full_bridge.FullBridge.__new__(full_bridge.FullBridge)
        bridge.plan=plan;bridge.key=self.key;bridge.cancelled=event;bridge.sequence=1;bridge.weight_dtype='int8'
        bridge.admission=FullRouteAdmission(plan);bridge.admission.admit_projection('public-test',16,2,1)
        tile=plan.schedule()[0];bridge.public_tile_digests={tile['id']:row['weightId']}
        bridge.remote_seconds=bridge.replay_seconds=0.;events=[]
        def exchange(*args,while_pending=None):
            events.append('exchange')
            if while_pending is not None:while_pending()
            bridge.sequence+=1
            return response
        def replay(*args):
            result=ext.cpu_matrix_multiplication_ntt(*args);events.append('replay')
            if cancel_during_replay:event.set()
            return result
        bridge.exchange=exchange;bridge.matrix_multiplication=replay;bridge.observe=lambda *args:events.append('verified')
        return bridge,tile,events

    def test_valid_but_wrong_encrypted_evaluation_rejects_before_response_deserialization(self):
        bridge,tile,events=self.bridge(self.wrong)
        with patch.object(ext,'CompressedResultEncryptedMatrix',SimpleNamespace(deserialize=lambda raw:self.fail('wrong native response deserialized'))):
            with self.assertRaises(ExecutionRejected) as failure:bridge.evaluate(self.cipher,self.weights,tile)
        self.assertEqual(failure.exception.code,'ciphertext-verification-failed');self.assertTrue(bridge.admission.failed)
        self.assertEqual(events,['exchange','replay'])

    def test_corrupt_bytes_and_cancelled_replay_never_reach_native_response_parser_or_retry(self):
        damaged=bytearray(self.good);damaged[len(damaged)//2]^=1
        for response,cancel in [(bytes(damaged),False),(self.good,True)]:
            event=threading.Event();bridge,tile,events=self.bridge(response,event,cancel)
            with patch.object(ext,'CompressedResultEncryptedMatrix',SimpleNamespace(deserialize=lambda raw:self.fail('unverified native response deserialized'))):
                with self.assertRaises(InterruptedError if cancel else ExecutionRejected):bridge.evaluate(self.cipher,self.weights,tile)
            self.assertEqual(events,['exchange','replay']);self.assertTrue(bridge.admission.failed)
            event.clear()
            with self.assertRaises(ValueError):bridge.evaluate(self.cipher,self.weights,tile)
            self.assertEqual(events,['exchange','replay'])

    def test_matching_native_response_is_deserialized_only_after_complete_replay(self):
        bridge,tile,events=self.bridge(self.good)
        def deserialize(raw):events.append('deserialize');return self.native_result_type.deserialize(raw)
        with patch.object(ext,'CompressedResultEncryptedMatrix',SimpleNamespace(deserialize=deserialize)):
            answer=bridge.evaluate(self.cipher,self.weights,tile)
        self.assertEqual(answer.serialize(),self.good);self.assertEqual(events,['exchange','replay','deserialize','verified'])
        bridge.admission.complete()

    def test_public_weight_corruption_rejects_before_any_disclosure_or_replay(self):
        bridge,tile,events=self.bridge(self.good);changed=self.weights.copy();changed[0,0]+=np.uint64(1)
        with self.assertRaises(ValueError):bridge.evaluate(self.cipher,changed,tile)
        self.assertEqual(events,[]);self.assertTrue(bridge.admission.failed)


if __name__=='__main__':unittest.main()
