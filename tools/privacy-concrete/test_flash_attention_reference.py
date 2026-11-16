"""Small public eager-QSA fixtures; no actual weights or encrypted attention claim."""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class QSAReferenceTests(unittest.TestCase):
    def fixture(self):
        import torch
        from flash_attention import ClientQSAAttention
        from flash_reference import load_reference
        from flash_request_state import ClientContext
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
        config=SimpleNamespace(hidden_size=4,num_attention_heads=2,num_key_value_heads=1,head_dim=4,
                               attention_dropout=0.,attention_bias=False,rms_norm_eps=1e-6,
                               indexer_n_heads=2,indexer_kv_heads=1,indexer_head_dim=2,
                               indexer_budget=2,indexer_compress_ratio=2,_attn_implementation='eager',max_position_embeddings=16)
        candidate=ClientQSAAttention(path,config,context_limit=8,rotary_width=2)
        with torch.no_grad():
            for index,parameter in enumerate(candidate.parameters()):
                parameter.copy_(torch.linspace(-.2,.3,parameter.numel()).reshape(parameter.shape)+(index+1)*.02)
        reference=load_reference(path,'attention')['Qwen4ExpTextAttention'](config,0).eval()
        reference.load_state_dict({key.replace('.source.','.'):value for key,value in candidate.model.state_dict().items()})
        context=ClientContext('tenant','privacy','public-qsa-model','key','request')
        return torch,candidate,reference,context

    def inputs(self,torch,length,past=0):
        angles=torch.arange(length,dtype=torch.float32).reshape(1,-1,1).expand(1,length,2)*.07
        visible=torch.arange(length)[None,:]<=torch.arange(past,length)[:,None]
        mask=torch.full((1,1,length-past,length),torch.finfo(torch.float32).min).masked_fill(visible[None,None],0)
        return (angles.cos(),angles.sin()),mask

    def test_prefill_and_three_cached_qsa_decode_steps_match_full_prefix_source(self):
        from flash_attention import ORDER
        from flash_attention_state import AttentionRequestState
        torch,candidate,reference,context=self.fixture();state=AttentionRequestState(context)
        values=torch.sin(torch.arange(28,dtype=torch.float32)*.17).reshape(1,7,4);trace=[]
        def project(operation,value):
            self.assertEqual(set(operation),{'id','inputWidth','outputWidth'})
            trace.append(operation['id']);return candidate.model.get_submodule(operation['id']).source(value)
        with torch.no_grad():
            for end in [4,5,6,7]:
                current=values[:,:4] if end==4 else values[:,end-1:end]
                positions,mask=self.inputs(torch,end,end-current.shape[1])
                actual=candidate(current,state,context,position_embeddings=positions,attention_mask=mask,
                                 execution='verified-projections',verified_client_projection=project)
                full_positions,full_mask=self.inputs(torch,end)
                expected,_=reference(values[:,:end],position_embeddings=full_positions,attention_mask=full_mask)
                torch.testing.assert_close(actual,expected[:,-current.shape[1]:],rtol=2e-4,atol=2e-5)
        self.assertEqual(trace,list(ORDER)*4);self.assertEqual(state.tokens,7)
        self.assertEqual(state.keys[(0,0)].shape[-2],7);self.assertEqual(state.indexer[(0,0)].shape[1],7)

    def test_qsa_token_selection_does_not_change_projection_route(self):
        from flash_attention import ORDER
        from flash_attention_state import AttentionRequestState
        torch,candidate,reference,context=self.fixture();selections=[];traces=[]
        for scale in [.17,-.29]:
            values=torch.sin(torch.arange(28,dtype=torch.float32)*scale).reshape(1,7,4)
            positions,mask=self.inputs(torch,7)
            with torch.no_grad():selections.append(reference.indexer(values,positions,mask,None))
            trace=[]
            def project(operation,value):trace.append(operation['id']);return candidate.model.get_submodule(operation['id']).source(value)
            candidate(values,AttentionRequestState(context),context,position_embeddings=positions,attention_mask=mask,
                      execution='verified-projections',verified_client_projection=project)
            traces.append(trace)
        self.assertEqual(traces,[list(ORDER),list(ORDER)])
        self.assertFalse(torch.equal(selections[0],selections[1]))
        self.assertTrue(all((selection[0,0,-1].sum()<=3) for selection in selections))

    def test_context_limit_wrong_namespace_and_invalid_masks_fail_before_projection(self):
        from flash_attention_state import AttentionRequestState
        torch,candidate,reference,context=self.fixture();calls=[]
        def forbidden(*args):calls.append(True);self.fail('invalid route disclosed a projection')
        values=torch.zeros((1,1,4));positions,mask=self.inputs(torch,1)
        for wrong_mask in [mask.bool(),torch.ones_like(mask),torch.full_like(mask,torch.finfo(torch.float32).min)]:
            with self.assertRaises(ValueError):candidate(values,AttentionRequestState(context),context,
                position_embeddings=positions,attention_mask=wrong_mask,execution='verified-projections',verified_client_projection=forbidden)
        with self.assertRaises(ValueError):candidate(values,AttentionRequestState(context),replace(context,key_id='other'),
            position_embeddings=positions,attention_mask=mask,execution='verified-projections',verified_client_projection=forbidden)
        positions,mask=self.inputs(torch,9)
        with self.assertRaises(ValueError):candidate(torch.zeros((1,9,4)),AttentionRequestState(context),context,
            position_embeddings=positions,attention_mask=mask,execution='verified-projections',verified_client_projection=forbidden)
        self.assertEqual(calls,[])

    def test_key_and_indexer_buffers_are_owned_by_separate_requests(self):
        from flash_attention_state import AttentionRequestState
        torch,candidate,reference,context=self.fixture();states=[AttentionRequestState(context),AttentionRequestState(replace(context,request_id='other',key_id='other'))]
        positions,mask=self.inputs(torch,4)
        for state in states:candidate(torch.ones((1,4,4)),state,state.context,position_embeddings=positions,attention_mask=mask,execution='public-reference')
        for field in ['keys','values','indexer']:
            left=getattr(states[0],field)[(0,0)];right=getattr(states[1],field)[(0,0)]
            self.assertNotEqual(left.data_ptr(),right.data_ptr());snapshot=right.clone();left.zero_()
            torch.testing.assert_close(right,snapshot,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
