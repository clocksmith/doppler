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
    def test_four_encrypted_feedback_steps_without_endpoint_coefficient_files(self):
        import numpy as np
        import concrete_ml_extensions as ext
        import flash_text_stream_plan as stream
        import full_bridge
        from checked_bridge import CheckedBridge
        from flash_component_plan import write_component
        from full_plan import tiles
        from public_weights import PreparedPublicTiles
        from linear_check import honest_norm_bound
        from protocol import digest
        from test_stream_transport import PipeProvider
        began=time.perf_counter();generator=np.random.default_rng(73)
        matrix=generator.integers(-11,12,size=(32,16),dtype=np.int64)
        public=matrix.astype('<i2').tobytes()
        row=dict(name='public-native-storage-probe',weightId=digest(public),inputWidth=32,outputWidth=16)
        geometry={row['name']:(32,16)};source={'schema':'synthetic-native-storage-probe/v1'}
        tile=tiles(row,2048,2048)[0];digests={tile['id']:digest(public)}
        with ExitStack() as stack:
            root=Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(patch.object(stream,'GEOMETRY',geometry));stack.enter_context(patch.object(stream,'TEXT',source))
            versions=stream.versions() # Real native source/binary/dependency validation.
            write_component(root,stream.CONFIG,[row],stream.semantics('d'*64,digests),versions,source=source)
            plan=stream.load_provider_plan(root)
            @contextmanager
            def coefficients(*args):yield io.BytesIO(public)
            binding=PreparedPublicTiles(plan,coefficients)
            parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(plan.definition()['parameters']['parameters']))
            secret,key=ext.cpu_create_private_key(parameters)
            children=[]
            def child(*args,**kwargs):
                result=PipeProvider(plan,str(root/'absent-provider-cache'));children.append(result);return result
            stack.enter_context(patch.object(full_bridge.subprocess,'Popen',side_effect=child))
            bridge=CheckedBridge(root,plan,key,backend_profile=stream.PROFILE,prepared_public_tiles=binding,
                parameter_profile=stream.CONFIG['parameterProfile'])
            stack.callback(bridge.close)
            values=generator.integers(0,2**12,size=(1,32),dtype=np.uint64);errors=[];greedy=[];durations=[]
            for step in range(4):
                started=time.perf_counter()
                with bridge.projection(row['name'],32,16,1):
                    encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=values)
                    bridge.stage_public_tile(matrix.astype(np.uint64),tile)
                    answer=bridge.evaluate(encrypted,matrix.astype(np.uint64),tile)
                    decoded=bridge.decode(answer,tile,values,lambda reply:ext.decrypt_matrix(reply,secret,parameters,16).astype(np.int64))
                expected=values.astype(np.int64)@matrix;deviation=decoded-expected
                self.assertLessEqual(np.linalg.norm(deviation),honest_norm_bound(16,bridge.check.noise,bridge.check.policy))
                self.assertEqual(int(decoded.argmax()),int(expected.argmax()))
                errors.append(int(np.abs(deviation).max()));token=int(decoded.argmax());greedy.append(token)
                # Private feedback stays client-side and is encrypted afresh.
                values=np.roll(values,1,axis=1).copy();values[0,token]=(values[0,token]+1)%4096
                durations.append(time.perf_counter()-started)
            bridge.finish();bridge.close();bridge.require_integrity()
            self.assertEqual(bridge.public_coefficient_bytes_sent,len(public)*4)
            self.assertEqual(list(root.iterdir()),[root/'plan.json']);self.assertIsNone(children[0].error)
            report=dict(schema='doppler.native-stream-storage-probe/v1',steps=4,greedyFeedbackMatches=True,
                nativeEncryptionSimulated=False,actualFlashCheckpointExecuted=False,separateMachine=False,
                elapsedSeconds=time.perf_counter()-began,stepSeconds=durations,maximumIntegerErrorByStep=errors,
                publicCoefficientBytesSent=bridge.public_coefficient_bytes_sent,totalFrameBytesSent=bridge.sent,
                totalFrameBytesReceived=bridge.received,evaluationKeyBytes=bridge.key_bytes,
                clientCoefficientDiskBytes=0,providerCoefficientDiskBytes=0,maxFramePayloadBytes=bridge.payload_limit,
                processPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                implementationVersions=versions,integrity=bridge.integrity_report(),productionBackendAvailable=False,securityQualified=False)
            output=os.environ.get('DOPPLER_STREAM_NATIVE_REPORT')
            if output:Path(output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':unittest.main()
