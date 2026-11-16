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
        from flash_prepared_graph import PreparedFlashGraph
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
            prepared_graph=PreparedFlashGraph(graph,executor,cache)
            prepared=prepared_graph._prepared
            plan=FullPlan(canonical(dict(steps=2,contextBucket=2,inputTileWidth=16,outputTileWidth=16,
                clientGraphId=graph.identity,parameters=dict(parameters=dict(bits_reserved_for_computation=27)),projections=prepared_graph.definitions())))
            bridge=full_bridge.FullBridge.__new__(full_bridge.FullBridge)
            bridge.plan=plan;bridge.admission=FullRouteAdmission(plan);bridge.cancelled=threading.Event();bridge.key=object();bridge.sequence=1
            bridge.key_id='key';bridge.session='request'
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
            tokens=[torch.tensor([[1,2]]),torch.tensor([[3]])]
            try:
                expected=[prepared_graph.reference(value,clear_state,context) for value in tokens]
                if failure=='binding':bridge.session='other-request'
                if failure=='plan':
                    definition=plan.definition();definition['clientGraphId']='other-graph';bridge.plan=FullPlan(canonical(definition))
                if failure=='source':source.closed=True
                if failure=='cache':
                    path=prepared[graph.order[0]].weight_path;raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
                with patch.object(full_model,'ext',native),patch.object(full_bridge,'ext',native):
                    if failure:
                        with self.assertRaises((ValueError,InterruptedError,RuntimeError)):
                            prepared_graph.execute(tokens[0],encrypted_state,context,bridge,object(),object())
                        self.assertEqual(calls['decrypt'],0)
                        with self.assertRaises((ValueError,InterruptedError)):
                            prepared_graph.execute(tokens[0],encrypted_state,context,bridge,object(),object())
                        if failure in ['cache','source','binding','plan']:self.assertEqual(calls['exchange'],0)
                        else:self.assertTrue(bridge.admission.failed);self.assertIsNone(bridge._projection_inputs)
                    else:
                        for value,reference in zip(tokens,expected):
                            actual=prepared_graph.execute(value,encrypted_state,context,bridge,object(),object())
                            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                        bridge.admission.complete();self.assertEqual(calls['replay'],len(plan.schedule())*2)
                        self.assertEqual(calls['exchange'],calls['replay']);self.assertEqual(calls['decrypt'],calls['replay'])
                        self.assertIsNone(bridge._projection_inputs)
            finally:
                clear_state.close();encrypted_state.close()
                self.assertTrue(all(p.secret is None and p.bridge is None and p.parameters is None for p in prepared.values()))

    def test_installed_route_builder_matches_complete_reference_operation_order(self):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_text_plan import geometry
        _,graph,parts,_=FlashModelReferenceTests().fixture()
        tensors={}
        def add(prefix,module):
            for name,value in module.state_dict().items():tensors[prefix+name]=dict(shape=list(value.shape),dtype='BF16')
        for i,layer in enumerate(parts[1]):add(f'model.language_model.layers.{i}.',layer)
        add('model.language_model.hyper_connection_mixer.',parts[2]);add('lm_head.',parts[3])
        actual=geometry(dict(tensors=tensors,layerCount=graph.config.num_hidden_layers,expertsPerLayer=graph.config.num_experts))
        self.assertEqual(list(actual),list(graph.order))
        for name,shape in actual.items():
            operation=graph.projections[name].operation
            self.assertEqual(shape,(operation['inputWidth'],operation['outputWidth']))

    def test_complete_two_steps_and_every_tile_replayed(self):self.run_case()
    def test_modified_reply_rejects_before_decryption(self):self.run_case('tamper')
    def test_cancelled_reply_rejects_before_decryption(self):self.run_case('cancel')
    def test_wrong_request_binding_rejects_before_disclosure(self):self.run_case('binding')
    def test_wrong_graph_plan_rejects_before_disclosure(self):self.run_case('plan')
    def test_closed_source_rejects_before_disclosure(self):self.run_case('source')
    def test_modified_cache_rejects_before_disclosure(self):self.run_case('cache')

if __name__=='__main__':unittest.main()
