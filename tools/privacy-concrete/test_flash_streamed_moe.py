"""Real small BF16 files/Torch math; not actual Flash-weight or encrypted evidence."""
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch/SafeTensors reference')
class FlashStreamedMoETests(unittest.TestCase):
    def fixture(self,directory,layer=47,config=None,moe=None):
        import hashlib
        import torch
        from torch import nn
        from safetensors.torch import save_file
        from flash_weights import FlashSource
        from flash_streamed_moe import SourceDenseClientMoE
        from flash_dense_moe import DenseClientMoE
        if config is None:config=SimpleNamespace(hidden_size=4,num_experts=3,num_experts_per_tok=1,moe_intermediate_size=3,
                               shared_expert_intermediate_size=2,norm_topk_prob=True)
        prefix=f'model.language_model.layers.{layer}.mlp.'
        h,e,m,s=config.hidden_size,config.num_experts,config.moe_intermediate_size,config.shared_expert_intermediate_size
        geometry={'shared_expert.gate_proj.weight':(s,h),'shared_expert.up_proj.weight':(s,h),'shared_expert.down_proj.weight':(h,s),
                  'gate.weight':(e,h),'shared_expert_gate.weight':(1,h),'experts.gate_up_proj':(e,2*m,h),'experts.down_proj':(e,h,m)}
        tensors={prefix+name:(torch.arange(__import__('math').prod(shape),dtype=torch.float32).reshape(shape)*.003-.06).to(torch.bfloat16)
                 for name,shape in geometry.items()}
        if moe is not None:
            tensors={prefix+name:(getattr(moe.experts,name.rsplit('.',1)[1]) if name.startswith('experts.') else moe.get_parameter(name)).detach().to(torch.bfloat16) for name in geometry}
        root=Path(directory);shard=root/'public.safetensors';save_file(tensors,str(shard))
        (root/'config.json').write_text('{}');(root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={name:shard.name for name in tensors})))
        def record(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        spec=dict(schema='synthetic-public-decoder-stage',files={name:record(root/name) for name in ['config.json','model.safetensors.index.json']},
                  shards={shard.name:record(shard)},tensors={name:dict(shard=shard.name,shape=list(value.shape),dtype='BF16') for name,value in tensors.items()})
        with patch('flash_weights.source_spec',return_value=spec):source=FlashSource(root)
        with patch('flash_streamed_moe.staging_spec',return_value=spec),patch.object(source,'tensor',wraps=source.tensor) as read:
            streamed=SourceDenseClientMoE(source,config,layer)
            read.assert_not_called()
        def linear(name):
            weight=tensors[prefix+name].float();value=nn.Linear(weight.shape[1],weight.shape[0],bias=False)
            value.weight=nn.Parameter(weight,requires_grad=False);return value
        reference=SimpleNamespace(shared_expert=SimpleNamespace(**{leaf+'_proj':linear('shared_expert.'+leaf+'_proj.weight') for leaf in ['gate','up','down']}),
            gate=linear('gate.weight'),shared_expert_gate=linear('shared_expert_gate.weight'),
            experts=SimpleNamespace(gate_up_proj=tensors[prefix+'experts.gate_up_proj'].float(),down_proj=tensors[prefix+'experts.down_proj'].float()))
        return torch,streamed,DenseClientMoE(reference,config),source,shard

    def test_fixed_all_expert_route_matches_materialized_reference_without_retained_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            torch,streamed,reference,source,_=self.fixture(directory)
            calls=[];leases=[];read=source.tensor
            def tensor(name,expert=None,**kwargs):
                value=read(name,expert=expert,**kwargs);calls.append((name,expert));leases.append(weakref.ref(value));return value
            with patch.object(source,'tensor',side_effect=tensor):
                for step in range(4):
                    value=torch.linspace(-.4,.7,8 if step==0 else 4).reshape(1,-1,4)
                    with torch.no_grad():actual=streamed(value);expected=reference(value)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    gc.collect();self.assertTrue(all(lease() is None for lease in leases))
            self.assertTrue(all(value.device.type=='meta' for value in streamed.parameters()))
            self.assertTrue(all(op.source_owner is streamed.source_owner for op in streamed.ops.values()))
            self.assertEqual(len(calls),4*(5+3*3))
            for step in range(4):
                experts=[index for name,index in calls[step*14:(step+1)*14] if '.experts.' in name]
                self.assertEqual(experts,[0,0,0,1,1,1,2,2,2])

    def test_closed_or_changed_stage_rejects_before_any_weight_read(self):
        for changed in ['closed','file','metadata','config']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                torch,streamed,_,source,shard=self.fixture(directory)
                if changed=='closed':source.closed=True
                elif changed=='file':shard.write_bytes(shard.read_bytes()+b'!')
                elif changed=='metadata':source.spec['schema']='changed'
                else:streamed.config.num_experts_per_tok=2
                with patch.object(source,'tensor',side_effect=AssertionError('changed source read')) as read:
                    with self.assertRaises(ValueError):streamed(torch.zeros((1,1,4)))
                read.assert_not_called()

    def test_geometry_mismatch_rejects_before_materialization(self):
        from flash_streamed_moe import SourceDenseClientMoE
        with tempfile.TemporaryDirectory() as directory:
            _,streamed,_,source,_=self.fixture(directory)
            streamed.config.hidden_size=5
            with patch('flash_streamed_moe.staging_spec',return_value=source.spec),patch.object(source,'tensor') as read:
                with self.assertRaises(ValueError):SourceDenseClientMoE(source,streamed.config,47)
                read.assert_not_called()

    @unittest.skipUnless(os.environ.get('DOPPLER_FLASH_SOURCE_DIR'),'pinned inspected full-graph source required')
    def test_source_bound_moe_integrates_with_complete_cached_autoregressive_graph(self):
        import copy
        from flash_model_graph import ClientFlashGraph
        from flash_request_state import ClientContext
        from test_flash_model_reference import FlashModelReferenceTests
        helper=FlashModelReferenceTests();torch,original,reference,_=helper.fixture()
        config=copy.deepcopy(original.config)
        embedding,layers,mixer,head=reference
        with tempfile.TemporaryDirectory() as directory:
            _,streamed,_,source,_=self.fixture(directory,layer=0,config=config,moe=layers[0].mlp)
            # Reference is the same BF16-rounded public MoE, with all other
            # tiny inspected-model parameters retained exactly.
            with torch.no_grad():
                for parameter in layers[0].mlp.parameters():parameter.copy_(parameter.to(torch.bfloat16).float())
            expected=copy.deepcopy((embedding,layers,mixer,head));layers[0].mlp=streamed
            model=ClientFlashGraph(original.reference_path,config,embedding,layers,mixer,head,context_limit=12)
            self.assertEqual(model.source_checks,(streamed.source_owner,))
            for mode in ['public-reference','verified-projections']:
                context=ClientContext('tenant','privacy',model.identity,'key',mode);state=model.new_state(context)
                tokens=torch.tensor([[1,2,0,3]],dtype=torch.int64)
                def project(operation,value):return model.projections[operation['id']].source(value)
                for step in range(4):
                    actual=model(tokens if step==0 else tokens[:,-1:],state,context,execution=mode,
                        verified_client_projection=None if mode=='public-reference' else project)
                    control=helper.full_prefix(torch,model,expected,tokens)
                    torch.testing.assert_close(actual[:,-1],control[:,-1],rtol=3e-4,atol=3e-5)
                    tokens=torch.cat([tokens,actual[:,-1].argmax(dim=-1,keepdim=True)],dim=-1)
                state.close()
            context=ClientContext('tenant','privacy',model.identity,'key','closed');state=model.new_state(context);source.closed=True
            with patch.object(model.embedding,'forward',side_effect=AssertionError('closed source reached embedding')),self.assertRaises(ValueError):
                model(tokens[:,:1],state,context,execution='verified-projections',verified_client_projection=lambda *args:self.fail('closed source disclosed'))
            self.assertTrue(state.failed)

    def test_verified_adapter_cannot_disclose_after_its_source_lease_closes(self):
        from flash_client_projections import ClientProjection,client_execution
        from flash_request_state import ClientContext,ClientRequestState
        class State(ClientRequestState):
            def clear_private_state(self):pass
        with tempfile.TemporaryDirectory() as directory:
            torch,streamed,_,source,_=self.fixture(directory)
            projection=ClientProjection('public',streamed.ops['shared_gate']);source.closed=True;calls=[]
            context=ClientContext('tenant','privacy','model','key','request');state=State(context)
            with self.assertRaises(ValueError):
                with client_execution(state,context,['public'],'verified-projections',lambda *args:calls.append(True),None):
                    projection(torch.zeros((1,1,4)))
            self.assertEqual(calls,[]);self.assertTrue(state.failed)


if __name__=='__main__':unittest.main()
