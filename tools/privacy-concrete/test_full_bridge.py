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
        self.bridge.matrix_multiplication=lambda *args:self.native.cpu_matrix_multiplication(*args)
        self.bridge.sequence=1;self.bridge.key=object();self.bridge.weight_dtype='int8'
        self.bridge.remote_seconds=self.bridge.replay_seconds=0.0
        self.bridge.observe=lambda *args:None
        def exchange(*args):self.exchanged.append(args);return b'invalid-circuit-output'
        self.bridge.exchange=exchange
        self.encrypted=SimpleNamespace(serialize=lambda:self.serialized.append(True) or b'ciphertext-only')
        self.matrix=SimpleNamespace(shape=(16,16),dtype='u64',view=lambda dtype:'pinned-public-weight')
        self.tile=dict(self.bridge.admission.next_tile())
        self.bridge.public_tile_digests={self.tile['id']:digest(b'pinned-public-weight')}

    def projection_route(self, input_width=16, output_width=48, steps=2):
        plan=SimpleNamespace(definition=lambda:dict(steps=steps,contextBucket=8),
            schedule=lambda:tiles(dict(name='first',weightId='public',inputWidth=input_width,outputWidth=output_width),16,16))
        self.bridge.admission=FullRouteAdmission(plan)
        self.bridge.public_tile_digests={tile['id']:digest(b'pinned-public-weight') for tile in plan.schedule()}
        self.replayed=[]
        def exchange(*args):
            self.exchanged.append(args);self.bridge.sequence+=1;return b'valid-circuit-output'
        self.bridge.exchange=exchange
        self.native.cpu_matrix_multiplication=lambda *args:self.replayed.append(args) or SimpleNamespace(serialize=lambda:b'valid-circuit-output')
        return plan.schedule()

    def test_projection_reuses_ciphertext_bytes_but_replays_every_output_tile(self):
        schedule=self.projection_route(output_width=16*75)
        for step in range(2):
            with self.bridge.projection('first',16,16*75,8 if step==0 else 1):
                for tile in schedule:self.bridge.evaluate(self.encrypted,self.matrix,tile)
                self.assertEqual(len(self.bridge._projection_inputs),1)
            self.assertIsNone(self.bridge._projection_inputs)
        self.bridge.admission.complete()
        self.assertEqual(len(self.serialized),2)
        self.assertEqual(len(self.exchanged),150);self.assertEqual(len(self.replayed),150)
        self.assertTrue(all(args[1]==b'ciphertext-only' for args in self.exchanged))

    def test_cached_input_cannot_bypass_changed_route_or_public_weight(self):
        for change in ['route','weight']:
            with self.subTest(change=change):
                self.setUp();schedule=self.projection_route()
                with self.assertRaises(ValueError):
                    with self.bridge.projection('first',16,48,8):
                        self.bridge.evaluate(self.encrypted,self.matrix,schedule[0])
                        retained=self.bridge._projection_inputs
                        if change=='route':tile=dict(schedule[1],id='changed')
                        else:
                            tile=schedule[1];self.matrix.view=lambda dtype:'changed-public-weight'
                        self.bridge.evaluate(self.encrypted,self.matrix,tile)
                self.assertEqual(retained,[]);self.assertIsNone(self.bridge._projection_inputs)
                self.assertEqual(len(self.serialized),1);self.assertEqual(len(self.exchanged),1)
                self.assertTrue(self.bridge.admission.failed)

    def test_payload_is_released_on_cancellation_transport_replay_and_projection_failure(self):
        for failure in ['cancel','transport','replay','projection']:
            with self.subTest(failure=failure):
                self.setUp();schedule=self.projection_route()
                with self.assertRaises((ValueError,InterruptedError,TimeoutError)):
                    with self.bridge.projection('first',16,48,8):
                        self.bridge.evaluate(self.encrypted,self.matrix,schedule[0])
                        retained=self.bridge._projection_inputs
                        if failure=='projection':raise ValueError('projection failed')
                        if failure=='cancel':
                            self.bridge.cancelled=threading.Event();self.bridge.cancelled.set()
                        if failure=='transport':
                            self.bridge.exchange=lambda *args:(_ for _ in ()).throw(TimeoutError('lost evaluator'))
                        if failure=='replay':self.bridge.exchange=lambda *args:b'wrong-ciphertext'
                        self.bridge.evaluate(self.encrypted,self.matrix,schedule[1])
                self.assertEqual(retained,[]);self.assertIsNone(self.bridge._projection_inputs)
                self.assertTrue(self.bridge.admission.failed)
                with self.assertRaises((ValueError,InterruptedError)):
                    self.bridge.evaluate(self.encrypted,self.matrix,schedule[1])
                self.assertEqual(len(self.serialized),1)

    def test_single_cached_payload_tracks_input_geometry_and_object_identity(self):
        schedule=self.projection_route(input_width=32,output_width=32,steps=1)
        other=SimpleNamespace(serialize=lambda:self.serialized.append(True) or b'different-ciphertext')
        with self.bridge.projection('first',32,32,8):
            for tile,encrypted in zip(schedule,[self.encrypted,self.encrypted,other,other]):
                self.bridge.evaluate(encrypted,self.matrix,tile)
                self.assertEqual(len(self.bridge._projection_inputs),1)
        self.assertEqual(len(self.serialized),4)
        self.assertEqual([args[1] for args in self.exchanged],[b'ciphertext-only']*2+[b'different-ciphertext']*2)

    def test_changed_ciphertext_or_key_cannot_reuse_cached_bytes(self):
        for changed in ['ciphertext','key']:
            self.setUp();schedule=self.projection_route(output_width=32,steps=1)
            with self.bridge.projection('first',16,32,8):
                self.bridge.evaluate(self.encrypted,self.matrix,schedule[0])
                encrypted=self.encrypted
                if changed=='key':self.bridge.key=object()
                else:encrypted=SimpleNamespace(serialize=lambda:self.serialized.append(True) or b'new-ciphertext')
                self.bridge.evaluate(encrypted,self.matrix,schedule[1])
            self.assertEqual(len(self.serialized),2)
            self.assertEqual(self.exchanged[-1][1],b'new-ciphertext' if changed=='ciphertext' else b'ciphertext-only')

    def test_cancellation_during_serialization_does_not_retain_or_disclose_bytes(self):
        self.projection_route();self.bridge.cancelled=threading.Event()
        def cancelled_payload():self.bridge.cancelled.set();return b'ciphertext-only'
        self.encrypted.serialize=cancelled_payload
        with self.assertRaises(InterruptedError):
            with self.bridge.projection('first',16,48,8):
                self.bridge.evaluate(self.encrypted,self.matrix,self.bridge.admission.next_tile())
        self.assertIsNone(self.bridge._projection_inputs);self.assertEqual(self.exchanged,[])
        self.assertTrue(self.bridge.admission.failed)

    def test_actual_full_linear_reuses_one_payload_and_releases_it_on_decrypt_failure(self):
        try:
            import numpy as np
            import torch
            import transformers
        except ImportError:self.skipTest('Torch/NumPy/Transformers adapter dependencies unavailable')
        from contextlib import nullcontext
        from full_plan import FullPlan,canonical
        spec=importlib.util.spec_from_file_location('_tested_full_model',Path(__file__).with_name('full_model.py'))
        model=importlib.util.module_from_spec(spec)
        serialized=[];exchanged=[];replayed=[];decrypted=[]
        native=SimpleNamespace(CompressedResultEncryptedMatrix=SimpleNamespace(deserialize=lambda raw:np.frombuffer(raw,dtype=np.int64).reshape(1,-1)))
        with patch.dict(sys.modules,{'full_bridge':self.module,'concrete_ml_extensions':native,
            'concrete.ml.quantization.linear_op_glwe_backend':SimpleNamespace(GLWELinearLayerExecutor=object)}):spec.loader.exec_module(model)
        self.module.np=np;self.module.ext=native
        for fail_decrypt in [False,True]:
            serialized.clear();exchanged.clear();replayed.clear();decrypted.clear()
            weights=np.arange(10,dtype=np.int8).reshape(2,5);value=torch.tensor([[3.,4.]])
            projection=model.FullLinear.__new__(model.FullLinear);torch.nn.Module.__init__(projection)
            projection.name='first';projection.definition=dict(name='first',weightId='public',inputWidth=2,outputWidth=5)
            projection.mode='execute';projection.context=None;projection.encoding=dict(quantizationBits=8)
            projection.scale=projection.zero_point=projection.sum_weights=None;projection.remote=SimpleNamespace(bias=None)
            projection.secret=object();projection.parameters=object();projection.coefficient_matrix=lambda:nullcontext(weights)
            projection.executor=SimpleNamespace(_dynamic_input_quantization=lambda *args:(value.long(),1,0,value.shape),
                _apply_correction_and_dequantize=lambda raw,*args:raw)
            bridge=self.module.FullBridge.__new__(self.module.FullBridge);projection.bridge=bridge
            bridge.plan=FullPlan(canonical(dict(steps=1,contextBucket=1,inputTileWidth=2,outputTileWidth=2,
                parameters=dict(parameters=dict(bits_reserved_for_computation=27)),projections=[projection.definition])))
            bridge.admission=FullRouteAdmission(bridge.plan);bridge.cancelled=None;bridge.key=object();bridge.sequence=1
            bridge.weight_dtype=np.int8;bridge.remote_seconds=bridge.replay_seconds=0.;bridge.observe=lambda *args:None
            schedule=bridge.plan.schedule()
            bridge.public_tile_digests={tile['id']:digest(np.ascontiguousarray(weights[:,tile['outputBegin']:tile['outputEnd']]).tobytes()) for tile in schedule}
            def encrypt(**kwargs):
                data=kwargs['data'];return SimpleNamespace(data=data,serialize=lambda:serialized.append(True) or data.tobytes())
            def replay(encrypted,matrix,key):
                replayed.append(True);return SimpleNamespace(serialize=lambda:(encrypted.data.view(np.int64)@matrix.view(np.int64)).tobytes())
            def exchange(kind,payload,response,tile_id):
                tile=schedule[bridge.admission.position];exchanged.append(payload);bridge.sequence+=1
                return (value.long().numpy()@weights[:,tile['outputBegin']:tile['outputEnd']].astype(np.int64)).tobytes()
            def decrypt(answer,*args):
                decrypted.append(True)
                if fail_decrypt and len(decrypted)==2:raise ValueError('simulated decrypt failure')
                return answer
            native.encrypt_matrix=encrypt;native.decrypt_matrix=decrypt;bridge.matrix_multiplication=replay;bridge.exchange=exchange
            if fail_decrypt:
                with self.assertRaises(ValueError):projection(value)
                self.assertTrue(bridge.admission.failed)
            else:
                actual=projection(value);bridge.admission.complete()
                torch.testing.assert_close(actual,torch.from_numpy(value.long().numpy()@weights.astype(np.int64)),rtol=0,atol=0)
            self.assertEqual(len(serialized),1);self.assertEqual(len(replayed),2 if fail_decrypt else 3)
            self.assertEqual(len(exchanged),len(replayed));self.assertTrue(all(payload==exchanged[0] for payload in exchanged))
            self.assertIsNone(bridge._projection_inputs)

    def test_early_scope_exit_and_nesting_fail_terminally(self):
        for nested in [False,True]:
            self.setUp();self.projection_route()
            with self.assertRaises(ValueError):
                with self.bridge.projection('first',16,48,8):
                    if nested:
                        with self.bridge.projection('first',16,48,8):pass
            self.assertIsNone(self.bridge._projection_inputs)
            self.assertTrue(self.bridge.admission.failed);self.assertEqual(self.serialized,[])

    def test_numerical_rejection_follows_the_same_complete_actual_pipe_transcript(self):
        import json
        import subprocess
        import tempfile
        from full_plan import FullPlan,canonical
        from verified_execution import complete_verified_route
        transcripts=[]
        for reject in [None,0,2,3]:
            with tempfile.TemporaryDirectory() as directory:
                trace=Path(directory)/'public-transcript.json';plan=FullPlan(canonical(dict(modelId='a'*64,steps=4,contextBucket=8,
                    operationTimeoutSeconds=5,inputTileWidth=16,outputTileWidth=16,
                    projections=[dict(name='first',weightId='b'*64,inputWidth=16,outputWidth=16)])))
                bridge=self.module.FullBridge.__new__(self.module.FullBridge)
                bridge.plan=plan;bridge.admission=FullRouteAdmission(plan);bridge.key=object();bridge.cancelled=None
                bridge.session='0'*32;bridge.key_id=digest(b'public-evaluation-fixture');bridge.key_bytes=25
                bridge.sequence=bridge.sent=bridge.received=0;bridge.remote_seconds=bridge.replay_seconds=0.
                bridge.payload_limit=None;bridge.weight_dtype='int8';bridge.observe=lambda *args:None
                bridge.matrix_multiplication=lambda *args:self.native.cpu_matrix_multiplication(*args)
                tile=plan.schedule()[0];bridge.public_tile_digests={tile['id']:digest(b'pinned-public-weight')}
                bridge.child=subprocess.Popen([sys.executable,str(Path(__file__).parent/'fixtures/framed_completion_provider.py'),plan.encoded.hex(),str(trace)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
                try:
                    bridge.exchange('evaluation-key',b'public-evaluation-fixture','ready')
                    def execute(step):
                        bridge.admission.admit_projection('first',16,16,8 if step==0 else 1)
                        bridge.evaluate(self.encrypted,self.matrix,tile)
                        return step
                    def validate(step,result):
                        self.assertEqual(bridge.child.poll(),0)
                        self.assertTrue(bridge.child.stdin.closed);self.assertTrue(bridge.child.stdout.closed)
                        if step==reject:raise ValueError('local numerical rejection')
                    if reject is None:self.assertEqual(complete_verified_route(bridge,4,execute,validate),[0,1,2,3])
                    else:
                        with self.assertRaises(ValueError):complete_verified_route(bridge,4,execute,validate)
                    transcripts.append(json.loads(trace.read_text()))
                    self.assertIsNone(bridge.key);self.assertTrue(bridge.admission.failed)
                finally:bridge.close()
        self.assertEqual(transcripts[0],['evaluation-key']+['evaluate']*4+['finish'])
        self.assertTrue(all(trace==transcripts[0] for trace in transcripts))

    def test_execution_metadata_is_immutable_and_not_reparsed_for_each_tile(self):
        from full_plan import FullPlan,canonical
        plan=FullPlan(canonical(dict(operationTimeoutSeconds=5,inputTileWidth=16,outputTileWidth=32,
            parameters=dict(parameters=dict(bits_reserved_for_computation=35)))))
        self.bridge.plan=plan
        with patch.object(FullPlan,'definition',autospec=True,side_effect=lambda self:__import__('json').loads(self.encoded)) as definition:
            for _index in range(100):
                self.assertEqual(self.bridge.operation_timeout,5)
                self.assertEqual(self.bridge.execution_geometry['arithmeticBits'],35)
            self.assertEqual(definition.call_count,2)
        with self.assertRaises(TypeError):self.bridge.execution_geometry['arithmeticBits']=64
        changed=plan.definition();changed['parameters']['parameters']['bits_reserved_for_computation']=64
        self.assertEqual(self.bridge.execution_geometry['arithmeticBits'],35)

    def test_close_is_terminal_before_payload_serialization_and_releases_evaluation_key(self):
        import io
        self.bridge.child=SimpleNamespace(poll=lambda:0,wait=lambda:0,stdin=io.BytesIO(),stdout=io.BytesIO())
        self.bridge.close();self.bridge.close()
        self.assertTrue(self.bridge.admission.failed);self.assertIsNone(self.bridge.key)
        self.assertTrue(self.bridge.child.stdin.closed);self.assertTrue(self.bridge.child.stdout.closed)
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(self.serialized,[]);self.assertEqual(self.exchanged,[])

    def test_cleanup_failure_still_terminates_attempt_and_releases_evaluation_material(self):
        import io
        def fail_wait():raise OSError('public process cleanup failure')
        self.bridge.child=SimpleNamespace(poll=lambda:0,wait=fail_wait,stdin=io.BytesIO(),stdout=io.BytesIO())
        with self.assertRaises(OSError):self.bridge.close()
        self.assertTrue(self.bridge.admission.failed);self.assertIsNone(self.bridge.key)
        self.assertTrue(self.bridge.child.stdin.closed);self.assertTrue(self.bridge.child.stdout.closed)
        with self.assertRaises(ValueError):self.bridge.evaluate(self.encrypted,self.matrix,self.tile)
        self.assertEqual(self.serialized,[]);self.assertEqual(self.exchanged,[])

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
