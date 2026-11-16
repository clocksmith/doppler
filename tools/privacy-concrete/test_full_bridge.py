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
from protocol import digest


class Plan:
    def definition(self):return dict(steps=4,contextBucket=8)
    def schedule(self):return tiles(dict(name='first',weightId='public',inputWidth=16,outputWidth=16))


class FullBridgeOrderingTests(unittest.TestCase):
    def setUp(self):
        self.serialized=[];self.exchanged=[];self.deserialized=[]
        native=SimpleNamespace(cpu_matrix_multiplication=lambda *args:SimpleNamespace(serialize=lambda:b'valid-circuit-output'),
                               CompressedResultEncryptedMatrix=SimpleNamespace(deserialize=lambda value:self.deserialized.append(value)))
        numpy=SimpleNamespace(uint64='u64',int64='i64',int8='i8',array_equal=lambda actual,expected:actual.encode()==expected.tobytes(),
                              ascontiguousarray=lambda value,dtype:SimpleNamespace(tobytes=lambda:value.encode()))
        spec=importlib.util.spec_from_file_location('_tested_full_bridge',Path(__file__).with_name('full_bridge.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'concrete_ml_extensions':native,'numpy':numpy}):spec.loader.exec_module(module)
        self.module=module
        self.bridge=module.FullBridge.__new__(module.FullBridge)
        self.bridge.admission=FullRouteAdmission(Plan())
        self.bridge.admission.admit_projection('first',16,16,8)
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
        self.bridge.public_tile_digests={self.tile['id']:digest(b'pinned-public-weight')}

    def test_unsupported_plan_rejects_before_key_export_or_provider_start(self):
        key=SimpleNamespace(serialize=lambda:self.fail('evaluation key exported'))
        with patch.object(self.module,'load_plan',side_effect=ValueError('unsupported model')), patch.object(self.module.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):self.module.FullBridge('unused',SimpleNamespace(identity='expected'),key)
            spawn.assert_not_called()

    def test_replaced_prepared_plan_rejects_before_key_export_or_provider_start(self):
        key=SimpleNamespace(serialize=lambda:self.fail('evaluation key exported'))
        with patch.object(self.module,'load_plan',return_value=SimpleNamespace(identity='replaced')), patch.object(self.module.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):self.module.FullBridge('unused',SimpleNamespace(identity='expected'),key)
            spawn.assert_not_called()

    def test_cache_change_during_admission_rejects_before_key_export_or_process(self):
        key=SimpleNamespace(serialize=lambda:self.fail('evaluation key exported'))
        plan=Plan();plan.identity='expected'
        with patch.object(self.module,'load_plan',return_value=plan), patch.object(self.module,'bind_public_tiles',side_effect=ValueError('cache changed')), patch.object(self.module.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):self.module.FullBridge('unused',plan,key)
            spawn.assert_not_called()

    def test_cancelled_constructor_does_not_prepare_or_export_key(self):
        event=threading.Event();event.set()
        key=SimpleNamespace(serialize=lambda:self.fail('evaluation key exported'))
        with patch.object(self.module,'load_plan') as load, patch.object(self.module.subprocess,'Popen') as spawn:
            with self.assertRaises(InterruptedError):self.module.FullBridge('unused',SimpleNamespace(identity='expected'),key,cancelled=event)
            load.assert_not_called();spawn.assert_not_called()

    def test_unallowed_remote_route_rejects_before_key_export_or_process(self):
        key=SimpleNamespace(serialize=lambda:self.fail('evaluation key exported'))
        plan=SimpleNamespace(identity='expected')
        deployment=SimpleNamespace(bind=lambda value:(_ for _ in ()).throw(ValueError('operator not allowed')))
        with patch.object(self.module,'load_plan',return_value=plan), patch.object(self.module.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):self.module.FullBridge('unused',plan,key,deployment=deployment)
            spawn.assert_not_called()

    def test_unknown_installed_backend_rejects_before_key_export_or_process(self):
        key=SimpleNamespace(serialize=lambda:self.fail('evaluation key exported'))
        with patch.object(self.module.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):self.module.FullBridge('unused',SimpleNamespace(identity='expected'),key,backend_profile='remote-advertised-full-flash')
            spawn.assert_not_called()

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
