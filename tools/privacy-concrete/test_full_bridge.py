"""Exercise actual client disclosure/replay ordering without loading native crypto or a model."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import patch
from full_admission import FullRouteAdmission
from full_plan import tiles


class Plan:
    def definition(self):return dict(steps=4,contextBucket=8)
    def schedule(self):return tiles(dict(name='first',weightId='public',inputWidth=16,outputWidth=16))


class Cache:
    def __getitem__(self,index):return 'pinned-public-weight'


class FullBridgeOrderingTests(unittest.TestCase):
    def setUp(self):
        self.serialized=[];self.exchanged=[];self.deserialized=[]
        native=SimpleNamespace(cpu_matrix_multiplication=lambda *args:SimpleNamespace(serialize=lambda:b'valid-circuit-output'),
                               CompressedResultEncryptedMatrix=SimpleNamespace(deserialize=lambda value:self.deserialized.append(value)))
        numpy=SimpleNamespace(uint64='u64',int64='i64',array_equal=lambda actual,expected:actual==expected)
        spec=importlib.util.spec_from_file_location('_tested_full_bridge',Path(__file__).with_name('full_bridge.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'concrete_ml_extensions':native,'numpy':numpy}):spec.loader.exec_module(module)
        self.bridge=module.FullBridge.__new__(module.FullBridge)
        self.bridge.admission=FullRouteAdmission(Plan())
        self.bridge.admission.admit_projection('first',16,16,8)
        self.bridge.public_weights={'first':Cache()}
        self.bridge.cancelled=None
        self.native=native
        self.bridge.sequence=1;self.bridge.key=object()
        self.bridge.remote_seconds=self.bridge.replay_seconds=0.0
        self.bridge.observe=lambda *args:None
        def exchange(*args):self.exchanged.append(args);return b'invalid-circuit-output'
        self.bridge.exchange=exchange
        self.encrypted=SimpleNamespace(serialize=lambda:self.serialized.append(True) or b'ciphertext-only')
        self.matrix=SimpleNamespace(shape=(16,16),dtype='u64',view=lambda dtype:'pinned-public-weight')
        self.tile=dict(self.bridge.admission.next_tile())

    def test_route_change_is_rejected_before_ciphertext_serialization(self):
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,dict(self.tile,id='replacement'))
        self.assertEqual(self.serialized,[]);self.assertEqual(self.exchanged,[])
        self.assertTrue(self.bridge.admission.failed)

    def test_weight_change_is_rejected_before_ciphertext_serialization(self):
        self.matrix.view=lambda dtype:'changed-public-weight'
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(self.serialized,[]);self.assertEqual(self.exchanged,[])
        self.assertTrue(self.bridge.admission.failed)

    def test_unverified_response_never_reaches_deserializer_and_attempt_is_terminal(self):
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(len(self.serialized),1);self.assertEqual(self.deserialized,[])
        self.assertEqual(self.exchanged[0][1],b'ciphertext-only')
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(len(self.serialized),1)

    def test_cancelled_attempt_cannot_serialize_or_resume_after_event_is_cleared(self):
        self.bridge.cancelled=threading.Event();self.bridge.cancelled.set()
        with self.assertRaises(InterruptedError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.bridge.cancelled.clear()
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(self.serialized,[]);self.assertEqual(self.exchanged,[])

    def test_cancellation_during_round_trip_stops_before_replay_or_deserialization(self):
        self.bridge.cancelled=threading.Event()
        def cancel(*args):self.bridge.cancelled.set();return b'valid-circuit-output'
        self.bridge.exchange=cancel
        self.native.cpu_matrix_multiplication=lambda *args:self.fail('replay started after cancellation')
        with self.assertRaises(InterruptedError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(self.deserialized,[]);self.assertTrue(self.bridge.admission.failed)

    def test_cancellation_during_native_replay_stops_before_deserialization(self):
        self.bridge.cancelled=threading.Event()
        self.bridge.exchange=lambda *args:b'valid-circuit-output'
        def cancel(*args):
            self.bridge.cancelled.set()
            return SimpleNamespace(serialize=lambda:b'valid-circuit-output')
        self.native.cpu_matrix_multiplication=cancel
        with self.assertRaises(InterruptedError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(self.deserialized,[]);self.assertTrue(self.bridge.admission.failed)

    def test_lost_evaluator_cannot_retry_or_deserialize(self):
        def disappear(*args):raise TimeoutError('evaluator disappeared')
        self.bridge.exchange=disappear
        with self.assertRaises(TimeoutError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(len(self.serialized),1);self.assertEqual(self.deserialized,[])


if __name__=='__main__':unittest.main()
