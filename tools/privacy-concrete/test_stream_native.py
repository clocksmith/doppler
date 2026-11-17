"""Opt-in real native GLWE tile streaming and four encrypted feedback steps.

Synthetic public matrix only, with the installed native/checking policy. No Flash
weights, full-model accuracy, remote host, production or security qualification.
"""
from contextlib import contextmanager,ExitStack
import io
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_STREAM_NATIVE')=='1','opt-in exact native storage probe')
class StreamNativeTests(unittest.TestCase):
    def run_case(self,*,rebuild=False,prefill5=False,full_tile=False):
        import numpy as np
        import concrete_ml_extensions as ext
        if prefill5:import flash_text_rebuild5_plan as stream
        elif rebuild:import flash_text_rebuild_plan as stream
        else:import flash_text_stream_plan as stream
        import full_bridge
        from checked_bridge import CheckedBridge
        from flash_component_plan import write_component
        from full_plan import tiles
        from public_weights import PreparedPublicTiles
        from linear_check import honest_norm_bound
        from protocol import digest
        from test_stream_transport import PipeProvider
        began=time.perf_counter();generator=np.random.default_rng(73)
        input_width,output_width=(2048,2048) if full_tile else (32,16)
        matrix=generator.integers(-11,12,size=(input_width,output_width),dtype=np.int64)
        with ExitStack() as stack:
            root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
            name='public-native-storage-probe';source_directory=None
            import torch
            from types import SimpleNamespace
            from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
            weight=torch.from_numpy(matrix.T.copy()).float().to(torch.bfloat16).float()
            context=SimpleNamespace(input_quantizers=[SimpleNamespace(quant_options=SimpleNamespace(n_bits=12))])
            quantized,*_=GLWELinearLayerExecutor(use_dynamic_quantization=True)._per_channel_weight_quantization(
                weight.numpy().T,context,torch.device('cpu'))
            matrix=quantized.numpy().astype(np.int64)
            if rebuild:
                from test_checkpoint_fixture import checkpoint_fixture
                source_directory=Path(stack.enter_context(tempfile.TemporaryDirectory()))
                specification,bindings=checkpoint_fixture(source_directory,{name:weight})
                stack.enter_context(patch('flash_provider_source.linear_source_spec',return_value=specification))
                stack.enter_context(patch('flash_provider_source.projection_bindings',return_value=bindings))
                stack.enter_context(patch.object(stream,'linear_source_spec',return_value=specification))
            public=matrix.astype('<i2').tobytes()
            row=dict(name=name,weightId=digest(public),inputWidth=input_width,outputWidth=output_width)
            geometry={name:(input_width,output_width)};source={'schema':'synthetic-native-storage-probe/v1'}
            tile=tiles(row,2048,2048)[0];digests={tile['id']:digest(public)}
            stack.enter_context(patch.object(stream,'GEOMETRY',geometry));stack.enter_context(patch.object(stream,'TEXT',source))
            versions=stream.versions() # Real native source/binary/dependency validation.
            write_component(root,stream.CONFIG,[row],stream.semantics('d'*64,digests),versions,source=source)
            plan=stream.load_provider_plan(root)
            @contextmanager
            def coefficients(*args):yield io.BytesIO(public)
            binding=PreparedPublicTiles(plan,coefficients)
            parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(plan.definition()['parameters']['parameters']))
            key_started=time.perf_counter();secret,key=ext.cpu_create_private_key(parameters);key_seconds=time.perf_counter()-key_started
            children=[]
            def child(*args,**kwargs):
                if rebuild:
                    from flash_provider_source import ProviderCheckpointSource
                    scratch=Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    checkpoint=ProviderCheckpointSource(source_directory,scratch,plan);stack.callback(checkpoint.close)
                    result=PipeProvider(plan,scratch,checkpoint_source=checkpoint);result.checkpoint=checkpoint;result.scratch=scratch
                else:result=PipeProvider(plan,str(root/'absent-provider-cache'))
                children.append(result);return result
            stack.enter_context(patch.object(full_bridge.subprocess,'Popen',side_effect=child))
            bridge=CheckedBridge(root,plan,key,backend_profile=stream.PROFILE,prepared_public_tiles=binding,
                parameter_profile=stream.CONFIG['parameterProfile'],
                **(dict(provider_source_directory=source_directory) if rebuild else {}))
            stack.callback(bridge.close)
            rows=5 if prefill5 else 1
            values=generator.integers(0,2**12,size=(rows,input_width),dtype=np.uint64);errors=[];greedy=[];durations=[]
            for step in range(4):
                started=time.perf_counter()
                with bridge.projection(row['name'],input_width,output_width,values.shape[0]):
                    encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=values)
                    if not rebuild:bridge.stage_public_tile(matrix.astype(np.uint64),tile)
                    answer=bridge.evaluate(encrypted,matrix.astype(np.uint64),tile)
                    decoded=bridge.decode(answer,tile,values,lambda reply:ext.decrypt_matrix(reply,secret,parameters,output_width).astype(np.int64))
                expected=values.astype(np.int64)@matrix;deviation=decoded-expected
                self.assertLessEqual(np.linalg.norm(deviation),honest_norm_bound(output_width,bridge.check.noise,bridge.check.policy))
                np.testing.assert_array_equal(decoded.argmax(axis=1),expected.argmax(axis=1))
                errors.append(int(np.abs(deviation).max()));token=int(decoded[-1].argmax());greedy.append(token)
                # Private feedback stays client-side and is encrypted afresh.
                values=np.roll(values[-1:],1,axis=1).copy();values[0,token]=(values[0,token]+1)%4096
                durations.append(time.perf_counter()-started)
            bridge.finish();bridge.close();bridge.require_integrity()
            self.assertEqual(bridge.public_coefficient_bytes_sent,0 if rebuild else len(public)*4)
            self.assertEqual(list(root.iterdir()),[root/'plan.json']);self.assertIsNone(children[0].error)
            report=dict(schema='doppler.native-stream-storage-probe/v1',steps=4,promptRows=rows,greedyFeedbackMatches=True,
                inputWidth=input_width,outputWidth=output_width,
                coefficientId=digest(public),syntheticInputSeed=73,
                nativeEncryptionSimulated=False,actualFlashCheckpointExecuted=False,separateMachine=False,
                elapsedSeconds=time.perf_counter()-began,stepSeconds=durations,maximumIntegerErrorByStep=errors,
                keyGenerationSeconds=key_seconds,evaluatorRoundTripSeconds=bridge.remote_seconds,
                clientSampledReplaySeconds=bridge.replay_seconds,clientCheckSeconds=bridge.check_seconds,
                publicCoefficientBytesSent=bridge.public_coefficient_bytes_sent,totalFrameBytesSent=bridge.sent,
                totalFrameBytesReceived=bridge.received,evaluationKeyBytes=bridge.key_bytes,
                clientCoefficientDiskBytes=0,providerCoefficientDiskBytes=0,maxFramePayloadBytes=bridge.payload_limit,
                processPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                implementationVersions=versions,integrity=bridge.integrity_report(),productionBackendAvailable=False,securityQualified=False)
            report['runtimeThreads']=dict(torch=torch.get_num_threads(),**{name:os.environ.get(name) for name in
                ('RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
            if rebuild:
                self.assertEqual(children[0].checkpoint.materializations,1)
                self.assertEqual(children[0].checkpoint.live_cache_bytes,0);self.assertEqual(list(children[0].scratch.iterdir()),[])
                report.update(providerMaterializations=children[0].checkpoint.materializations,
                    providerPeakCoefficientDiskBytes=children[0].checkpoint.peak_cache_bytes,
                    providerSourceBytesVerified=children[0].checkpoint.source_bytes_verified,
                    providerSourceVerificationSeconds=children[0].checkpoint.source_verification_seconds,
                    providerMaterializationSeconds=children[0].checkpoint.materialization_seconds,
                    publicCoefficientStorage=plan.definition()['publicCoefficientStorage'])
            output=os.environ.get('DOPPLER_STREAM_NATIVE_REPORT')
            if output:
                path=Path(output)
                if full_tile:path=path.with_name('checkpoint-rebuild-prefill5-full-tile-native.json')
                elif prefill5:path=path.with_name('checkpoint-rebuild-prefill5-native.json')
                elif rebuild:path=path.with_name('checkpoint-rebuild-native.json')
                path.write_text(json.dumps(report,indent=2)+'\n')
            return report

    def test_four_encrypted_feedback_steps_without_endpoint_coefficient_files(self):self.run_case()
    def test_verified_checkpoint_cache_sends_zero_coefficients_over_four_native_steps(self):self.run_case(rebuild=True)
    def test_five_row_prefill_then_single_row_feedback_on_explicit_native_profile(self):self.run_case(rebuild=True,prefill5=True)
    @unittest.skipUnless(os.environ.get('DOPPLER_STREAM_NATIVE_FULL_TILE')=='1','opt-in full-sized profile tile')
    def test_full_sized_five_row_profile_tile_then_single_row_feedback(self):self.run_case(rebuild=True,prefill5=True,full_tile=True)

if __name__=='__main__':unittest.main()
