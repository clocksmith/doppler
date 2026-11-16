"""Tiny complete graph, real quantization and replay, simulated crypto/transport.
Synthetic plan only: no installed full Flash plan, native or model qualification.
"""
import os,tempfile,threading,unittest
from types import SimpleNamespace
from unittest.mock import patch

@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch adapter')
class FlashEncryptedGraphAdapterTests(unittest.TestCase):
    def run_case(self,failure=None):
        import numpy as np
        import torch
        import full_model,full_bridge
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from flash_decoder_source import bind_decoder
        from flash_model_graph import ClientFlashGraph
        from flash_request_state import ClientContext
        from test_flash_decoder_source import DecoderSourceTests
        from test_flash_model_reference import FlashModelReferenceTests
        from full_plan import FullPlan,canonical
        from full_admission import FullRouteAdmission
        from protocol import digest
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as cache:
            _,original,_,source,spec=DecoderSourceTests().fixture(directory,0)
            with patch('flash_decoder_source.staging_spec',return_value=spec),patch('flash_streamed_moe.staging_spec',return_value=spec),patch('flash_decoder_source.resolve_text_config',return_value=original.config):
                layer,config=bind_decoder(source,original.reference_path,0)
            _,_,parts,_=FlashModelReferenceTests().fixture();parts[1][0]=layer
            graph=ClientFlashGraph(original.reference_path,config,*parts,context_limit=12)
            executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
            prepared={name:full_model.FullLinear(name,graph.projections[name].source,executor,cache) for name in graph.order}
            for projection in prepared.values():projection.release_float_reference()
            plan=FullPlan(canonical(dict(steps=2,contextBucket=2,inputTileWidth=16,outputTileWidth=16,
                parameters=dict(parameters=dict(bits_reserved_for_computation=27)),projections=[prepared[name].definition for name in graph.order])))
            bridge=full_bridge.FullBridge.__new__(full_bridge.FullBridge)
            bridge.plan=plan;bridge.admission=FullRouteAdmission(plan);bridge.cancelled=threading.Event();bridge.key=object();bridge.sequence=1
            bridge.weight_dtype=np.int8;bridge.remote_seconds=bridge.replay_seconds=0.;bridge.observe=lambda *args:None
            bridge.public_tile_digests={};weights={}
            for name,projection in prepared.items():
                with projection.coefficient_matrix() as matrix:weights[name]=matrix.copy()
            for tile in plan.schedule():
                block=weights[tile['name']][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']]
                bridge.public_tile_digests[tile['id']]=digest(np.ascontiguousarray(block).tobytes())
            calls=dict(replay=0,exchange=0,decrypt=0);active={}
            def encrypt(**kwargs):
                data=kwargs['data'];active['rows']=data.shape[0]
                return SimpleNamespace(data=data,serialize=lambda:data.tobytes())
            def replay(encrypted,matrix,key):
                calls['replay']+=1
                return SimpleNamespace(serialize=lambda:(encrypted.data.view(np.int64)@matrix.view(np.int64)).tobytes())
            def exchange(kind,payload,response,tile_id,while_pending):
                calls['exchange']+=1;tile=bridge.admission.next_tile()
                inputs=np.frombuffer(payload,dtype=np.uint64).reshape(active['rows'],-1).view(np.int64)
                matrix=weights[tile['name']][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']].astype(np.int64)
                answer=(inputs@matrix).tobytes();while_pending();bridge.sequence+=1
                if failure=='cancel':bridge.cancelled.set()
                if failure=='tamper':answer=bytes([answer[0]^1])+answer[1:]
                return answer
            def decrypt(answer,*args):calls['decrypt']+=1;return answer
            native=SimpleNamespace(encrypt_matrix=encrypt,decrypt_matrix=decrypt,
                CompressedResultEncryptedMatrix=SimpleNamespace(deserialize=lambda raw:np.frombuffer(raw,dtype=np.int64).reshape(active['rows'],-1)))
            bridge.matrix_multiplication=replay;bridge.exchange=exchange
            context=ClientContext('tenant','privacy',graph.identity,'key','request')
            clear_state=graph.new_state(context);encrypted_state=graph.new_state(context)
            projections={name:projection.prepared_copy() for name,projection in prepared.items()}
            def project(operation,value):return projections[operation['id']](value)
            tokens=[torch.tensor([[1,2]]),torch.tensor([[3]])]
            try:
                expected=[graph(value,clear_state,context,execution='verified-projections',verified_client_projection=project) for value in tokens]
                for projection in projections.values():projection.mode='execute';projection.bridge=bridge;projection.secret=object();projection.parameters=object()
                if failure=='source':source.closed=True
                if failure=='cache':
                    path=projections[graph.order[0]].weight_path;raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
                with patch.object(full_model,'ext',native),patch.object(full_bridge,'ext',native):
                    if failure:
                        with self.assertRaises((ValueError,InterruptedError,RuntimeError)):
                            graph(tokens[0],encrypted_state,context,execution='verified-projections',verified_client_projection=project,cancelled=bridge.cancelled)
                        self.assertEqual(calls['decrypt'],0)
                        with self.assertRaises(ValueError):
                            graph(tokens[0],encrypted_state,context,execution='verified-projections',verified_client_projection=project)
                        if failure in ['cache','source']:self.assertEqual(calls['exchange'],0)
                        else:self.assertTrue(bridge.admission.failed);self.assertIsNone(bridge._projection_inputs)
                    else:
                        for value,reference in zip(tokens,expected):
                            actual=graph(value,encrypted_state,context,execution='verified-projections',verified_client_projection=project)
                            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                        bridge.admission.complete();self.assertEqual(calls['replay'],len(plan.schedule())*2)
                        self.assertEqual(calls['exchange'],calls['replay']);self.assertEqual(calls['decrypt'],calls['replay'])
                        self.assertIsNone(bridge._projection_inputs)
            finally:
                clear_state.close();encrypted_state.close()
                for projection in projections.values():projection.secret=projection.bridge=projection.parameters=None

    def test_complete_two_steps_and_every_tile_replayed(self):self.run_case()
    def test_modified_reply_rejects_before_decryption(self):self.run_case('tamper')
    def test_cancelled_reply_rejects_before_decryption(self):self.run_case('cancel')
    def test_closed_source_rejects_before_disclosure(self):self.run_case('source')
    def test_modified_cache_rejects_before_disclosure(self):self.run_case('cache')

if __name__=='__main__':unittest.main()
