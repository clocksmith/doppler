"""Tiny local-table PLE source comparisons; not the actual 51B-parameter ngram table."""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class PLEReferenceTests(unittest.TestCase):
    def fixture(self):
        import torch
        from flash_ple import ClientPLE
        from flash_reference import load_reference
        from flash_request_state import ClientContext
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
        config=SimpleNamespace(hidden_size=2,hc_count=3,ple_embed_dim=8,ngram_size=3,heads_per_ngram=2,
                               vocab_size=17,ngram_vocab_size_base=101,seed=19,eos_token_id=0,
                               make_ngram_vocab_size_divisible_by=8,ple_conv_kernel_size=2,rms_norm_eps=1e-6,max_position_embeddings=16)
        candidate=ClientPLE(path,config,context_limit=8,embedding_limit_bytes=1024*1024)
        with torch.no_grad():
            for index,parameter in enumerate(candidate.parameters()):
                parameter.copy_(torch.linspace(-.2,.3,parameter.numel()).reshape(parameter.shape)+(index+1)*.01)
        reference=load_reference(path,'ple')['Qwen4ExpTextPLELayer'](config,1,0).eval()
        reference.load_state_dict({key.replace('.source.','.'):value for key,value in candidate.model.state_dict().items()})
        context=ClientContext('tenant','privacy','public-ple-model','key','request')
        return torch,candidate,reference,context,config,path

    def test_prefill_and_four_cached_updates_cross_eos_match_full_prefix(self):
        from flash_ple import ORDER
        from flash_ple_state import PLERequestState
        torch,candidate,reference,context,config,path=self.fixture();state=PLERequestState(context)
        values=torch.sin(torch.arange(42,dtype=torch.float32)*.11).reshape(1,7,6)
        tokens=torch.tensor([[1,2,3,0,4,5,6]],dtype=torch.int64);trace=[]
        def project(operation,value):trace.append(operation['id']);return getattr(candidate.model,operation['id']).source(value)
        with torch.no_grad():
            for end in [3,4,5,6,7]:
                start=0 if end==3 else end-1
                actual=candidate(values[:,start:end],tokens[:,start:end],state,context,execution='verified-projections',verified_client_projection=project)
                expected=reference(values[:,:end],tokens[:,:end],cache=None)[:,start:end]
                torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
        self.assertEqual(trace,list(ORDER)*5);self.assertEqual(state.conv[(1,2)].tolist(),[[5,6]])
        self.assertEqual(set(candidate.model.ple_embedding._buffers),{'layer_multipliers','ngram_heads_vocab_sizes','ngram_heads_offsets'})

    def test_embedding_lookup_and_private_history_stay_local_before_projection(self):
        from flash_ple_state import PLERequestState
        torch,candidate,reference,context,config,path=self.fixture();state=PLERequestState(context);lookups=[];projections=[]
        handle=candidate.model.ple_embedding.ngram_embedding.register_forward_pre_hook(lambda module,args:lookups.append(True))
        try:
            def forbidden(*args):projections.append(True);self.fail('invalid context projected private tokens')
            with self.assertRaises(ValueError):candidate(torch.zeros((1,1,6)),torch.tensor([[1]]),state,replace(context,key_id='other'),
                execution='verified-projections',verified_client_projection=forbidden)
            self.assertEqual(lookups,[]);self.assertEqual(projections,[])
            with self.assertRaises(ValueError):candidate(torch.zeros((1,1,6)),torch.tensor([[1]]),PLERequestState(context),context,execution='verified-projections')
            self.assertEqual(lookups,[])
        finally:handle.remove()

    def test_materialization_budget_rejects_before_allocating_a_large_table(self):
        from flash_ple import ClientPLE
        torch,candidate,reference,context,config,path=self.fixture()
        config.ngram_vocab_size_base=1000000
        with self.assertRaises(ValueError):ClientPLE(path,config,context_limit=8,embedding_limit_bytes=1024*1024)

    def test_two_request_histories_do_not_alias(self):
        from flash_ple_state import PLERequestState
        torch,candidate,reference,context,config,path=self.fixture()
        states=[PLERequestState(context),PLERequestState(replace(context,request_id='other',key_id='other'))]
        for state in states:candidate(torch.zeros((1,3,6)),torch.tensor([[1,2,3]]),state,state.context,execution='public-reference')
        for slot in [(1,1),(1,2)]:
            left=states[0].conv[slot];right=states[1].conv[slot];snapshot=right.clone()
            self.assertNotEqual(left.data_ptr(),right.data_ptr());left.zero_();torch.testing.assert_close(right,snapshot,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
