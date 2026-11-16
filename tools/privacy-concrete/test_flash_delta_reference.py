"""Small public FP32 prefill/decode comparisons; no actual Flash weights or FHE claims."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class DeltaReferenceTests(unittest.TestCase):
    def fixture(self):
        import torch
        from flash_delta import ClientDeltaNet
        from flash_delta_state import DeltaContext
        from flash_reference import load_reference
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
        config=SimpleNamespace(hidden_size=4,linear_num_value_heads=2,linear_num_key_heads=1,
                               linear_key_head_dim=2,linear_value_head_dim=2,linear_conv_kernel_dim=3,
                               hidden_act='silu',rms_norm_eps=1e-6,output_gate_type='sigmoid',layer_types=['linear_attention'])
        candidate=ClientDeltaNet(path,config)
        with torch.no_grad():
            for index,parameter in enumerate(candidate.parameters()):
                parameter.copy_(torch.linspace(-.2,.3,parameter.numel()).reshape(parameter.shape)+(index+1)*.01)
        reference=load_reference(path,'delta')['Qwen4ExpTextGatedDeltaNet'](config,0).eval()
        reference.load_state_dict({key.replace('.source.','.'):value for key,value in candidate.model.state_dict().items()})
        context=DeltaContext('tenant-a','privacy','public-fixture-model','key-a','request-a')
        return torch,candidate,reference,context

    def test_prefill_and_three_cached_decode_steps_match_full_prefix_source(self):
        from flash_delta_state import DeltaRequestState
        from flash_delta import ORDER
        torch,candidate,reference,context=self.fixture();state=DeltaRequestState(context)
        values=torch.sin(torch.arange(24,dtype=torch.float32)*.13).reshape(1,6,4);trace=[]
        def project(operation,value):
            trace.append(operation['id']);return getattr(candidate.model,operation['id']).source(value)
        with torch.no_grad():
            for end in [3,4,5,6]:
                current=values[:,:3] if end==3 else values[:,end-1:end]
                actual=candidate(current,state,context,execution='verified-projections',verified_client_projection=project)
                expected=reference(values[:,:end])[:,-current.shape[1]:]
                torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
        self.assertEqual(trace,list(ORDER)*4);self.assertEqual(state.tokens,6)

    def test_concurrent_requests_share_public_weights_and_isolate_private_states(self):
        from flash_delta_state import DeltaRequestState
        torch,candidate,reference,context=self.fixture()
        inputs=[torch.sin(torch.arange(20,dtype=torch.float32)*scale).reshape(1,5,4) for scale in [.13,.27]]
        contexts=[context,replace(context,tenant_id='tenant-b',key_id='key-b',request_id='request-b')]
        states=[DeltaRequestState(value) for value in contexts]
        def execute(index):
            with torch.no_grad():
                candidate(inputs[index][:,:3],states[index],contexts[index],execution='public-reference')
                return candidate(inputs[index][:,3:],states[index],contexts[index],execution='public-reference')
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(execute,[0,1]))
        with torch.no_grad():
            for value,result in zip(inputs,results):torch.testing.assert_close(result,reference(value)[:,-2:],rtol=2e-4,atol=2e-5)
        self.assertNotEqual(states[0].conv[(0,0)].data_ptr(),states[1].conv[(0,0)].data_ptr())
        self.assertNotEqual(states[0].recurrent[(0,0)].data_ptr(),states[1].recurrent[(0,0)].data_ptr())
        snapshot=states[1].recurrent[(0,0)].clone();states[0].recurrent[(0,0)].zero_()
        torch.testing.assert_close(states[1].recurrent[(0,0)],snapshot,rtol=0,atol=0)

    def test_namespace_change_and_mid_projection_cancellation_stop_before_next_hop(self):
        from flash_delta_state import DeltaRequestState
        torch,candidate,reference,context=self.fixture();value=torch.zeros((1,1,4));calls=[]
        def forbidden(*args):calls.append(True);self.fail('context failure disclosed a projection')
        state=DeltaRequestState(context)
        with self.assertRaises(ValueError):candidate(value,state,replace(context,key_id='other'),execution='verified-projections',verified_client_projection=forbidden)
        self.assertEqual(calls,[])
        state=DeltaRequestState(context);event=threading.Event()
        def cancelled(operation,value):
            calls.append(operation['id']);result=getattr(candidate.model,operation['id']).source(value);event.set();return result
        with self.assertRaises(InterruptedError):candidate(value,state,context,execution='verified-projections',verified_client_projection=cancelled,cancelled=event)
        self.assertEqual(calls,['in_proj_qkv']);self.assertEqual(state.conv,{})
        event.clear()
        with self.assertRaises(ValueError):candidate(value,state,context,execution='verified-projections',verified_client_projection=forbidden)
        self.assertEqual(calls,['in_proj_qkv'])

    def test_missing_projection_backend_never_falls_back_to_reference(self):
        from flash_delta_state import DeltaRequestState
        torch,candidate,reference,context=self.fixture();state=DeltaRequestState(context)
        with self.assertRaises(ValueError):candidate(torch.zeros((1,1,4)),state,context,execution='verified-projections')
        self.assertTrue(state.failed)


if __name__=='__main__':unittest.main()
