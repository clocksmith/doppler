"""Complete mixed-layer autoregressive client graph; tiny public weights, not Flash model evidence."""
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import os
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import threading
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class FlashModelReferenceTests(unittest.TestCase):
    def fixture(self):
        import torch
        from torch import nn
        from flash_reference import load_reference
        from flash_model_graph import ClientFlashGraph
        from flash_request_state import ClientContext
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py';source=load_reference(path,'decoder')
        config=SimpleNamespace(hidden_size=4,num_hidden_layers=3,layer_types=['linear_attention','indexed_attention','linear_attention'],
            num_experts=3,num_experts_per_tok=2,moe_intermediate_size=3,shared_expert_intermediate_size=3,intermediate_size=3,
            hidden_act='silu',norm_topk_prob=True,hc_count=2,hc_lowrank=3,rms_norm_eps=1e-6,
            linear_num_value_heads=2,linear_num_key_heads=1,linear_key_head_dim=2,linear_value_head_dim=2,linear_conv_kernel_dim=3,output_gate_type='sigmoid',
            num_attention_heads=2,num_key_value_heads=1,head_dim=4,attention_dropout=0.,attention_bias=False,indexer_n_heads=2,indexer_kv_heads=1,
            indexer_head_dim=2,indexer_budget=2,indexer_compress_ratio=2,_attn_implementation='eager',max_position_embeddings=16,
            rope_parameters=dict(rope_type='default',rope_theta=10000,partial_rotary_factor=.5,mrope_section=[1,0,0]),
            ple_layer_ids=[2],ple_embed_dim=8,ngram_size=3,heads_per_ngram=2,vocab_size=17,ngram_vocab_size_base=101,seed=19,eos_token_id=0,
            make_ngram_vocab_size_divisible_by=8,ple_conv_kernel_size=2)
        embedding=nn.Embedding(17,4);layers=nn.ModuleList([source['Qwen4ExpTextDecoderLayer'](config,i) for i in range(3)])
        mixer=source['Qwen4ExpTextGatedResidual'](config,use_combine=False);head=nn.Linear(4,17,bias=False)
        with torch.no_grad():
            for index,parameter in enumerate(nn.ModuleList([embedding,layers,mixer,head]).parameters()):
                parameter.copy_(torch.linspace(-.08,.11,parameter.numel()).reshape(parameter.shape)+(index+1)*.001)
        reference=copy.deepcopy((embedding,layers,mixer,head))
        model=ClientFlashGraph(path,config,embedding,layers,mixer,head,context_limit=12)
        context=ClientContext('tenant','privacy',model.identity,'key','request')
        return torch,model,reference,context

    def full_prefix(self,torch,model,reference,tokens):
        from flash_rotary import text_rotary
        embedding,layers,mixer,head=reference;total=tokens.shape[1]
        positions=text_rotary(model.reference_path,model.config,total)
        visible=torch.arange(total)[None,:]<=torch.arange(total)[:,None]
        mask=torch.full((1,1,total,total),torch.finfo(torch.float32).min).masked_fill(visible[None,None],0)
        with torch.no_grad():
            hidden=embedding(tokens).repeat(1,1,model.config.hc_count)
            for layer in layers:hidden=layer(hidden,position_embeddings=positions,attention_mask=mask,conv_mask=None,cache=None,ple_input_ids=tokens)
            return head(mixer(hidden))

    def test_complete_four_token_autoregression_matches_sparse_full_prefix_source(self):
        torch,model,reference,context=self.fixture();state=model.new_state(context);tokens=torch.tensor([[1,2,0,3]],dtype=torch.int64);trace=[];generated=[]
        def projection(operation,value):trace.append(operation['id']);return model.projections[operation['id']].source(value)
        for step in range(4):
            current=tokens if step==0 else tokens[:,-1:]
            actual=model(current,state,context,execution='verified-projections',verified_client_projection=projection)
            expected=self.full_prefix(torch,model,reference,tokens)
            torch.testing.assert_close(actual[:,-1],expected[:,-1],rtol=3e-4,atol=3e-5)
            token=int(actual[0,-1].argmax());self.assertEqual(token,int(expected[0,-1].argmax()))
            generated.append(token);tokens=torch.cat([tokens,torch.tensor([[token]],dtype=torch.int64)],dim=1)
        self.assertEqual(len(generated),4);self.assertEqual(trace,list(model.order)*4);self.assertEqual(state.tokens,7)
        self.assertEqual(state.ple[1].tokens,7);self.assertTrue(state.attention[0].recurrent);self.assertTrue(state.attention[1].indexer)
        self.assertEqual(len([name for name in model.order if '.mlp.ops.expert_' in name]),3*3*3)

    def test_two_requests_share_public_weights_and_isolate_all_actual_cache_stores(self):
        torch,model,reference,context=self.fixture();contexts=[context,replace(context,tenant_id='other',key_id='other',request_id='other')]
        states=[model.new_state(c) for c in contexts];tokens=[torch.tensor([[1,2,0,3,4]],dtype=torch.int64),torch.tensor([[5,6,0,7,8]],dtype=torch.int64)]
        def execute(i):
            model(tokens[i][:,:4],states[i],contexts[i],execution='public-reference')
            return model(tokens[i][:,4:],states[i],contexts[i],execution='public-reference')
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(execute,[0,1]))
        for i in range(2):torch.testing.assert_close(results[i][:,-1],self.full_prefix(torch,model,reference,tokens[i])[:,-1],rtol=3e-4,atol=3e-5)
        for pair in [(states[0].attention[0].conv,states[1].attention[0].conv),
                     (states[0].attention[0].recurrent,states[1].attention[0].recurrent),
                     (states[0].attention[1].keys,states[1].attention[1].keys),
                     (states[0].attention[1].indexer,states[1].attention[1].indexer),
                     (states[0].ple[1].conv,states[1].ple[1].conv)]:
            for key in pair[0]:self.assertNotEqual(pair[0][key].data_ptr(),pair[1][key].data_ptr())

    def test_invalid_client_intermediate_never_reaches_projection_callback(self):
        torch,model,reference,context=self.fixture();calls=[]
        def forbidden(operation,value):calls.append(operation['id']);raise AssertionError('invalid client intermediate disclosed')
        for value in [torch.ones((1,1,3)),torch.full((1,1,4),float('nan'))]:
            with patch.object(model.embedding,'forward',return_value=value):
                with self.assertRaises(ValueError):model(torch.tensor([[1]],dtype=torch.int64),model.new_state(context),context,
                    execution='verified-projections',verified_client_projection=forbidden)
        self.assertEqual(calls,[])

    def test_cancellation_namespace_and_context_budget_fail_terminally_before_later_hops(self):
        torch,model,reference,context=self.fixture();tokens=torch.tensor([[1,2,0,3]],dtype=torch.int64);state=model.new_state(context);event=threading.Event();calls=[]
        def project(operation,value):
            calls.append(operation['id']);result=model.projections[operation['id']].source(value)
            if operation['id']=='layers.0.mlp.ops.expert_0_up':event.set()
            return result
        with self.assertRaises(InterruptedError):model(tokens,state,context,execution='verified-projections',verified_client_projection=project,cancelled=event)
        self.assertEqual(calls[-1],'layers.0.mlp.ops.expert_0_up');self.assertTrue(state.failed)
        for child in state.children():
            self.assertTrue(child.failed)
            for name in ['conv','recurrent','keys','values','indexer']:
                if hasattr(child,name):self.assertEqual(getattr(child,name),{})
        event.clear();before=list(calls)
        with self.assertRaises(ValueError):model(tokens,state,context,execution='verified-projections',verified_client_projection=project)
        for candidate,request in [(model.new_state(context),replace(context,key_id='other')),(model.new_state(context),context)]:
            invalid=tokens if request!=context else torch.ones((1,13),dtype=torch.int64)
            with self.assertRaises(ValueError):model(invalid,candidate,request,execution='verified-projections',verified_client_projection=project)
        self.assertEqual(calls,before)


if __name__=='__main__':unittest.main()
