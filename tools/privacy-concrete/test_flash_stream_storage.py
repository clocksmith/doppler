"""Actual quantizer, evicted coefficients, real framed pipes and checked bridge.

Tiny patched source geometry only. Native encryption is simulated with public
fixture integers; this is storage/integration evidence, never cryptographic or
actual-checkpoint qualification.
"""
from contextlib import ExitStack
import os
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in real Torch quantizer')
class FlashStreamStorageTests(unittest.TestCase):
    def fixture(self,stack,*,rebuild=False,bf16_control=False,prefill5=False):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_prepared_graph import PreparedFlashGraph,EPHEMERAL_STORAGE
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        if prefill5:import flash_text_rebuild5_plan as stream
        elif rebuild:import flash_text_rebuild_plan as stream
        else:import flash_text_stream_plan as stream
        torch,graph,parts,_=FlashModelReferenceTests().fixture()
        if rebuild or bf16_control:
            import copy
            from flash_model_graph import ClientFlashGraph
            with torch.no_grad():
                for parameter in torch.nn.ModuleList(list(parts)).parameters():parameter.copy_(parameter.to(torch.bfloat16).float())
            graph=ClientFlashGraph(graph.reference_path,copy.deepcopy(graph.config),*parts,context_limit=12)
        if rebuild:
            from test_checkpoint_fixture import checkpoint_fixture
            raw=Path(stack.enter_context(tempfile.TemporaryDirectory()))
            spec,bindings=checkpoint_fixture(raw,{name:graph.projections[name].source.weight for name in graph.order})
            self.provider_source_directory=raw
            stack.enter_context(patch('flash_provider_source.linear_source_spec',return_value=spec))
            stack.enter_context(patch('flash_provider_source.projection_bindings',return_value=bindings))
            stack.enter_context(patch.object(stream,'linear_source_spec',return_value=spec))
        root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        prepared=PreparedFlashGraph(graph,GLWELinearLayerExecutor(use_dynamic_quantization=True),root,
            quantization_bits=12,storage=EPHEMERAL_STORAGE,output_chunk_rows=3)
        self.assertEqual(list(root.iterdir()),[])
        geometry={row['name']:(row['inputWidth'],row['outputWidth']) for row in prepared.definitions()}
        stack.enter_context(patch.object(stream,'GEOMETRY',geometry))
        stack.enter_context(patch.object(stream,'TEXT',{'schema':'synthetic-complete-graph'}))
        stack.enter_context(patch.object(stream,'versions',return_value={'test-fixture':'simulated-native'}))
        plan,binding=stream.write_plan(root,prepared)
        self.assertEqual(list(root.iterdir()),[root/'plan.json'])
        return torch,graph,prepared,root,plan,binding

    def run_case(self,failure=None,*,rebuild=False,bf16_control=False,generation=False,prefill5=False,bounded_float=False):
        import numpy as np
        import full_model,full_bridge,full_provider,checked_bridge
        from flash_request_state import ClientContext
        from plan_backends import STREAMED_PROFILE,REBUILD_PROFILE,REBUILD_PREFILL5_PROFILE
        PROFILE=REBUILD_PREFILL5_PROFILE if prefill5 else REBUILD_PROFILE if rebuild else STREAMED_PROFILE
        from full_status import ExecutionRejected
        from full_plan import canonical
        from protocol import digest
        import struct
        with ExitStack() as stack:
            torch,graph,prepared,root,plan,binding=self.fixture(stack,rebuild=rebuild,bf16_control=bf16_control,prefill5=prefill5)
            counts=dict(parsed=0,decrypted=0,evaluated=0)
            class Cipher:
                def __init__(self,data):self.data=np.ascontiguousarray(data,dtype=np.uint64)
                def serialize(self):return struct.pack('!II',*self.data.shape)+self.data.tobytes()
                @staticmethod
                def deserialize(raw):
                    counts['parsed']+=1
                    rows,columns=struct.unpack('!II',raw[:8]);return Cipher(np.frombuffer(raw[8:],dtype=np.uint64).reshape(rows,columns))
            class Key:
                def serialize(self):return b'public-fixture-evaluation-key'
                @staticmethod
                def deserialize(raw):
                    if raw!=b'public-fixture-evaluation-key':raise ValueError('wrong evaluation key')
                    return Key()
            def evaluate(encrypted,matrix,key):
                result=encrypted.data.view(np.int64)@matrix.view(np.int64)
                if threading.current_thread() is not threading.main_thread():
                    counts['evaluated']+=1
                    if failure=='reply' and counts['evaluated']==1:result[0,0]+=2**26
                return SimpleNamespace(serialize=lambda:result.tobytes())
            def decrypt(raw,secret,parameters,width):
                counts['decrypted']+=1
                return np.frombuffer(raw,dtype=np.int64).reshape(-1,width)
            native=SimpleNamespace(encrypt_matrix=lambda **kw:Cipher(kw['data']),
                decrypt_matrix=decrypt,
                CpuCompressionKey=Key,EncryptedMatrix=Cipher,
                CompressedResultEncryptedMatrix=SimpleNamespace(deserialize=lambda raw:raw))
            for module in (full_model,full_bridge,full_provider,checked_bridge):stack.enter_context(patch.object(module,'ext',native))
            for module in (full_bridge,full_provider):
                stack.enter_context(patch.object(module,'matrix_evaluator',return_value=evaluate))
                stack.enter_context(patch.object(module,'prepare_evaluator_key',return_value=None))
            children=[]
            from test_stream_transport import PipeProvider
            def child(*args,**kwargs):
                if rebuild:
                    from flash_provider_source import ProviderCheckpointSource
                    scratch=Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    checkpoint=ProviderCheckpointSource(self.provider_source_directory,scratch,plan);stack.callback(checkpoint.close)
                    result=PipeProvider(plan,scratch,checkpoint_source=checkpoint)
                    result.checkpoint=checkpoint;result.scratch=scratch
                else:result=PipeProvider(plan,str(root/'no-provider-cache'))
                children.append(result);return result
            stack.enter_context(patch.object(full_bridge.subprocess,'Popen',side_effect=child))
            bridge=checked_bridge.CheckedBridge(root,plan,Key(),backend_profile=PROFILE,prepared_public_tiles=binding,
                parameter_profile='glwe-quant12-message35-storage48/v1',cancelled=threading.Event(),
                **(dict(provider_source_directory=self.provider_source_directory) if rebuild else {}))
            stack.callback(bridge.close)
            if failure=='public':
                serialize=bridge._serialize_input
                def corrupt(*args):
                    raw=serialize(*args);return raw[:9]+bytes([raw[9]^1])+raw[10:]
                bridge._serialize_input=corrupt
            elif failure=='source':
                with torch.no_grad():graph.projections[graph.order[0]].source.weight.add_(1)
            context=ClientContext('tenant','restricted',graph.identity,bridge.key_id,bridge.session)
            clear_context=ClientContext('tenant','restricted',graph.identity,'reference-key','reference-request')
            state=graph.new_state(context);clear=graph.new_state(clear_context);stack.callback(state.close);stack.callback(clear.close)
            prompt=[1,2,3,4,5] if prefill5 else [1]
            tokens=torch.tensor([prompt]);generated=[]
            parameters=SimpleNamespace(serialize=lambda:json.dumps(plan.definition()['parameters']['parameters']))
            if failure=='parameters':parameters=SimpleNamespace(serialize=lambda:'{}')
            if failure in ('public','source','parameters'):
                with self.assertRaises(ValueError):prepared.execute(tokens,state,context,bridge,object(),parameters)
                self.assertTrue(state.failed);self.assertTrue(bridge.admission.failed)
                self.assertEqual(counts['parsed'],0);self.assertEqual(counts['decrypted'],0)
                self.assertEqual(list(root.iterdir()),[root/'plan.json'])
                return
            if generation:
                from flash_generation import generate
                from dataclasses import replace
                from flash_float_reference import BoundedFloatingReference,REFERENCE_ID
                floating=None if not bounded_float else BoundedFloatingReference(graph,cancelled=bridge.cancelled,output_chunk_rows=2)
                if floating is not None:stack.callback(floating.close)
                observed_states=[];original_new_state=graph.new_state;reference_calls=[]
                def new_state(request):
                    result=original_new_state(request);observed_states.append(result);return result
                original_reference=prepared.reference
                def reference(*args,**kwargs):
                    self.assertIsNotNone(children[0].poll(),'reference acceptance leaked before provider exit')
                    reference_calls.append(args[2])
                    result=original_reference(*args,**kwargs)
                    if failure=='numerical':
                        result=result.clone();result[...,int(result[0,-1].argmax())]=-1e10
                    return result
                with patch.object(graph,'new_state',side_effect=new_state),patch.object(prepared,'reference',side_effect=reference):
                    if failure in ('reply','numerical','prefill','context','cancel','used'):
                        if failure=='cancel':bridge.cancelled.set()
                        if failure=='used':
                            bridge.admission.position=len(plan.schedule());bridge.sequence+=len(plan.schedule())
                        request=replace(context,request_id='other-request') if failure=='context' else context
                        error=InterruptedError if failure=='cancel' else ValueError if failure in ('prefill','context','used') else ExecutionRejected
                        with self.assertRaises(error):
                            generate(prepared,[1,2] if failure=='prefill' else prompt,bridge,object(),parameters,request)
                    else:
                        result=generate(prepared,prompt,bridge,object(),parameters,context,floating_reference=floating)
                        self.assertEqual(len(result['clientLocalOutputTokens']),4)
                        self.assertEqual(result['promptRows'],len(prompt))
                        self.assertTrue(result['matchesTeacherForcedQuantizedGreedyReference'])
                        self.assertEqual(result['maxLogitErrors'],[0.]*4)
                        self.assertFalse(result['productionBackendAvailable']);self.assertFalse(result['securityQualified'])
                        if bounded_float:self.assertEqual(result['floatingReference'],REFERENCE_ID)
                self.assertIsNotNone(children[0].poll());self.assertTrue(all(state.failed for state in observed_states))
                self.assertEqual(len({state.context for state in observed_states}),len(observed_states))
                self.assertEqual(counts['evaluated'],0 if failure in ('prefill','context','cancel','used') else len(plan.schedule())*4)
                self.assertEqual(len(reference_calls),0 if failure in ('reply','prefill','context','cancel','used') else 1 if failure=='numerical' else 4)
                self.assertEqual(list(root.iterdir()),[root/'plan.json'])
                self.assertEqual(list(children[0].scratch.iterdir()),[])
                return
            rematerialization_seconds=[]
            for step in range(4):
                expected=prepared.reference(tokens,clear,clear_context)
                before=prepared.rematerialization_seconds
                actual=prepared.execute(tokens,state,context,bridge,object(),parameters)
                rematerialization_seconds.append(prepared.rematerialization_seconds-before)
                if failure is None:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                tokens=actual[:,-1].argmax(dim=-1,keepdim=True);generated.append(int(tokens.item()))
                self.assertEqual(list(root.iterdir()),[root/'plan.json'])
                self.assertIsNone(bridge._public_tile_payload)
            bridge.finish();bridge.close()
            if failure=='reply':
                with self.assertRaises(ExecutionRejected):bridge.require_integrity()
            else:bridge.require_integrity()
            self.assertEqual(len(generated),4);self.assertEqual(counts['evaluated'],len(plan.schedule())*4)
            self.assertEqual(bridge.public_coefficient_bytes_sent,plan.definition()['publicCoefficientBytesPerRoute'])
            self.assertGreater(bridge.sent,bridge.public_coefficient_bytes_sent)
            self.assertEqual(len(rematerialization_seconds),4);self.assertTrue(all(value>0 for value in rematerialization_seconds))
            self.assertFalse((root/'no-provider-cache').exists());self.assertIsNone(children[0].error)
            report=dict(schema='doppler.tiny-flash-stream-storage-test/v1',projectionCount=len(prepared.order),
                clientGraphId=graph.identity,coefficientInventoryId=digest(canonical(prepared.definitions())),
                encryptedTiles=len(plan.schedule())*4,steps=4,publicCoefficientBytesSent=bridge.public_coefficient_bytes_sent,
                totalFrameBytesSent=bridge.sent,totalFrameBytesReceived=bridge.received,
                clientInitialPreparationSeconds=prepared.preparation_seconds,
                clientRematerializationSecondsByStep=rematerialization_seconds,
                clientLargestCoefficientFileBytes=max(row['inputWidth']*row['outputWidth']*2 for row in prepared.definitions()),
                providerCoefficientDiskBytes=0,maxFramePayloadBytes=bridge.payload_limit,
                nativeEncryptionSimulated=True,actualFlashCheckpointExecuted=False,productionBackendAvailable=False,securityQualified=False)
            if rebuild:
                self.assertEqual(children[0].checkpoint.live_cache_bytes,0);self.assertEqual(list(children[0].scratch.iterdir()),[])
                report.update(providerMaterializations=children[0].checkpoint.materializations,
                    providerPeakCoefficientDiskBytes=children[0].checkpoint.peak_cache_bytes,
                    providerSourceBytesVerified=children[0].checkpoint.source_bytes_verified,
                    providerMaterializationSeconds=children[0].checkpoint.materialization_seconds,
                    providerSourceVerificationSeconds=children[0].checkpoint.source_verification_seconds)
            return report

    def test_ephemeral_reference_and_complete_checked_pipe_route_match_four_tokens(self):self.run_case()
    def test_bad_public_bytes_reject_before_provider_ciphertext_parse(self):self.run_case('public')
    def test_changed_source_rejects_before_private_payload(self):self.run_case('source')
    def test_wrong_actual_parameters_reject_before_private_payload(self):self.run_case('parameters')
    def test_corrupt_result_defers_rejection_until_complete_route_closed(self):self.run_case('reply')
    def test_verified_checkpoint_reconstruction_sends_zero_coefficients_across_four_steps(self):self.run_case(rebuild=True)

    def test_fixed_generation_validates_after_close_and_cleans_separate_request_states(self):self.run_case(rebuild=True,generation=True)
    def test_fixed_generation_integrity_failure_returns_no_output_or_reference_feedback(self):self.run_case('reply',rebuild=True,generation=True)
    def test_fixed_generation_numerical_rejection_follows_all_four_remote_steps(self):self.run_case('numerical',rebuild=True,generation=True)
    def test_fixed_generation_prefill_mismatch_closes_before_any_ciphertext(self):self.run_case('prefill',rebuild=True,generation=True)
    def test_fixed_generation_wrong_request_closes_before_any_ciphertext(self):self.run_case('context',rebuild=True,generation=True)
    def test_fixed_generation_cancelled_before_start_closes_without_ciphertext(self):self.run_case('cancel',rebuild=True,generation=True)
    def test_fixed_generation_cannot_restart_a_partly_consumed_route(self):self.run_case('used',rebuild=True,generation=True)
    def test_five_row_profile_completes_fixed_generation_without_prompt_truncation(self):self.run_case(rebuild=True,generation=True,prefill5=True)
    def test_five_row_profile_rejects_a_shorter_prompt_before_ciphertext(self):self.run_case('prefill',rebuild=True,generation=True,prefill5=True)
    def test_five_row_generation_validates_with_bounded_floating_oracle_after_close(self):self.run_case(rebuild=True,generation=True,prefill5=True,bounded_float=True)

    def test_one_and_five_row_profiles_cannot_substitute_manifests_or_ssh_routes(self):
        from plan_backends import REBUILD_PROFILE,REBUILD_PREFILL5_PROFILE,compiled_backend
        from ssh_route import SSHRoute,ENDPOINT_FIELDS
        from test_ssh_route import fixture
        with ExitStack() as stack:
            _,graph1,_,root1,plan1,binding1=self.fixture(stack,rebuild=True)
            _,graph5,_,root5,plan5,binding5=self.fixture(stack,rebuild=True,prefill5=True)
            self.assertEqual(graph1.identity,graph5.identity)
            self.assertEqual(plan1.definition()['contextBucket'],1);self.assertEqual(plan5.definition()['contextBucket'],5)
            self.assertNotEqual(plan1.identity,plan5.identity);self.assertNotEqual(plan1.protocol_id,plan5.protocol_id)
            for profile,root,binding in ((REBUILD_PROFILE,root5,binding5),(REBUILD_PREFILL5_PROFILE,root1,binding1)):
                with self.assertRaises(ValueError):compiled_backend(profile).load_client_plan(root,binding)
            _,value=fixture();endpoint={key:value[key] for key in ENDPOINT_FIELDS};endpoint['remotePublicSource']='/srv/pinned-public-source'
            route=SSHRoute.for_compiled_checkpoint(root5,binding5,endpoint,('research-provider',),backend_profile=REBUILD_PREFILL5_PROFILE)
            self.assertEqual(route.definition()['backendProfile'],REBUILD_PREFILL5_PROFILE)
            self.assertEqual(route.bind(plan5,('research-provider',)).definition(),plan5.definition())
            with self.assertRaises(ValueError):route.bind(plan1,('research-provider',))
            with self.assertRaises(ValueError):SSHRoute.for_compiled_checkpoint(root5,binding5,endpoint,(),backend_profile=REBUILD_PREFILL5_PROFILE)
            with self.assertRaises(ValueError):SSHRoute.for_compiled_checkpoint(root5,binding5,endpoint,('research-provider',))

    def test_same_bf16_model_measures_coefficient_traffic_reduction_across_four_steps(self):
        streamed=self.run_case(bf16_control=True);rebuilt=self.run_case(rebuild=True)
        self.assertEqual(streamed['clientGraphId'],rebuilt['clientGraphId'])
        self.assertEqual(streamed['coefficientInventoryId'],rebuilt['coefficientInventoryId'])
        self.assertGreater(streamed['publicCoefficientBytesSent'],0);self.assertEqual(rebuilt['publicCoefficientBytesSent'],0)
        self.assertLess(rebuilt['totalFrameBytesSent'],streamed['totalFrameBytesSent'])

    def test_manifest_alone_cannot_admit_client_or_expand_legacy_route(self):
        from flash_text_stream_plan import load_client_plan,load_provider_plan,PROFILE
        from plan_backends import load_backend_plan
        from full_bridge import FullBridge
        from checked_bridge import CheckedBridge
        from ssh_route import SSHRoute,ENDPOINT_FIELDS,validate
        from test_ssh_route import fixture
        with ExitStack() as stack:
            _,_,_,root,plan,binding=self.fixture(stack)
            self.assertEqual(load_provider_plan(root),plan)
            for fake in (None,{'validated':True},dict(binding.entries)):
                key=Mock()
                with patch('full_bridge.subprocess.Popen') as spawn:
                    with self.assertRaises(ValueError):CheckedBridge(root,plan,key,backend_profile=PROFILE,prepared_public_tiles=fake,
                        parameter_profile='glwe-quant12-message35-storage48/v1')
                    key.serialize.assert_not_called();spawn.assert_not_called()
            with self.assertRaises(ValueError):load_backend_plan(root,PROFILE)
            _,endpoint=fixture();endpoint={key:endpoint[key] for key in ENDPOINT_FIELDS}
            with self.assertRaises(ValueError):SSHRoute.for_compiled_stream(root,binding,endpoint,())
            route=SSHRoute.for_compiled_stream(root,binding,endpoint,('research-provider',))
            self.assertEqual(route.bind(plan,('research-provider',)).definition(),plan.definition())
            value=route.definition();value['schema']='doppler.encrypted-ssh-check-route/v1'
            with self.assertRaises(ValueError):validate(value)

    def test_full_installed_metadata_route_fits_manifest_and_declares_wire_cost(self):
        import flash_text_stream_plan as stream
        from flash_component_plan import write_component
        from full_plan import tiles
        from protocol import digest
        definitions=[dict(name=name,inputWidth=width,outputWidth=height,weightId=digest(name.encode()))
            for name,(width,height) in stream.GEOMETRY.items()]
        tile_digests={tile['id']:'a'*64 for row in definitions for tile in tiles(row,2048,2048)}
        with tempfile.TemporaryDirectory() as directory,patch.object(stream,'versions',return_value={'fixture':'metadata-only'}):
            meaning=stream.semantics('b'*64,tile_digests)
            write_component(directory,stream.CONFIG,definitions,meaning,stream.versions(),source=stream.TEXT)
            self.assertLess((Path(directory)/'plan.json').stat().st_size,stream.MAX_MANIFEST_BYTES)
            plan=stream.load_provider_plan(directory)
            self.assertEqual(len(definitions),74501);self.assertEqual(len(plan.schedule()),150868)
            self.assertEqual(meaning['publicCoefficientBytesPerStep'],250210795520)
            self.assertEqual(meaning['publicCoefficientBytesPerRoute'],1000843182080)
            self.assertFalse(plan.definition()['productionBackendAvailable'])

    def test_atomic_manifest_reuse_corruption_and_cancellation_preserve_siblings(self):
        from flash_text_stream_plan import write_plan
        with ExitStack() as stack:
            _,_,prepared,root,plan,binding=self.fixture(stack)
            path=root/'plan.json';sibling=root/'keep';sibling.write_text('keep')
            before=path.stat().st_ino
            self.assertEqual(write_plan(root,prepared)[0],plan);self.assertEqual(path.stat().st_ino,before)
            event=threading.Event();event.set()
            with self.assertRaises(InterruptedError):write_plan(root,prepared,cancelled=event)
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):write_plan(root,prepared)
            self.assertEqual(path.read_bytes(),b'corrupt');self.assertEqual(sibling.read_text(),'keep')
            self.assertEqual({item.name for item in root.iterdir()},{'plan.json','keep'})

    def test_rematerialization_cancellation_and_changed_public_metadata_clean_only_owned_cache(self):
        from protocol import require_active
        with ExitStack() as stack:
            _,_,prepared,root,_,_=self.fixture(stack)
            sibling=root/'keep';sibling.write_text('keep');name=prepared.order[0]
            event=threading.Event();quantize=prepared.executor._per_channel_weight_quantization
            def cancel(*args):
                result=quantize(*args);event.set();return result
            with patch.object(prepared.executor,'_per_channel_weight_quantization',side_effect=cancel):
                with self.assertRaises(InterruptedError):
                    with prepared.projection_lease(name,cancelled=event):self.fail('cancelled coefficient lease escaped')
            self.assertEqual({path.name for path in root.iterdir()},{'plan.json','keep'})
            prepared._prepared[name].scale+=1
            with self.assertRaises(ValueError):
                with prepared.projection_lease(name):self.fail('changed quantizer metadata escaped')
            self.assertEqual(sibling.read_text(),'keep');self.assertFalse(list(root.glob('flash-public-projection-*')))

if __name__=='__main__':unittest.main()
