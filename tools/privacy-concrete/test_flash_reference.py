"""Opt-in pinned official-source component comparison; public small FP32 fixtures, not Flash weights."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class FlashReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torch import nn
        from torch.nn import functional as F
        cls.torch=torch
        from flash_reference import load_reference
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
        cls.classes=load_reference(path,'moe')

    def fixture(self):
        from oblivious_experts import ExpertProgram,FlashMoEProgram
        torch=self.torch
        config=SimpleNamespace(num_experts=3,num_experts_per_tok=2,hidden_size=2,moe_intermediate_size=2,
                               shared_expert_intermediate_size=2,intermediate_size=2,hidden_act='silu',norm_topk_prob=True)
        model=self.classes['Qwen4ExpTextSparseMoeBlock'](config).eval()
        with torch.no_grad():
            for index,parameter in enumerate(model.parameters()):
                parameter.copy_(torch.linspace(-.4,.6,parameter.numel()).reshape(parameter.shape)+(index+1)*.03)
        spec=FlashMoEProgram(ExpertProgram('a'*64,3,2,2,2),2,True,'b'*64)
        def project(operation,values):
            name=operation['id'];value=torch.tensor(values,dtype=torch.float32)
            if name.startswith('shared/'):
                layer=name.split('/')[1]
                matrix=model.shared_expert_gate.weight if layer=='mix' else getattr(model.shared_expert,layer+'_proj').weight
            elif name=='router/logits':matrix=model.gate.weight
            else:
                _,expert,layer=name.split('/');expert=int(expert)
                if layer=='down':matrix=model.experts.down_proj[expert]
                else:matrix=model.experts.gate_up_proj[expert][0:2] if layer=='gate' else model.experts.gate_up_proj[expert][2:4]
            return torch.nn.functional.linear(value,matrix).tolist()
        return model,spec,project

    def test_complete_shared_router_and_dense_experts_match_official_sparse_block(self):
        torch=self.torch;model,spec,project=self.fixture();traces=[];selections=[]
        with torch.no_grad():
            for hidden in [[.4,-.2],[-.5,.9],[1.2,.3],[-.4,-.6]]:
                values=torch.tensor(hidden)
                selections.append(tuple(model.gate(values.reshape(1,2))[2].reshape(-1).tolist()))
                expected=model(values.reshape(1,1,2)).reshape(-1)
                trace=[]
                def observed(operation,values):trace.append(dict(operation));return project(operation,values)
                actual=spec.attempt().run(hidden,observed)
                torch.testing.assert_close(torch.tensor(actual),expected,rtol=1e-5,atol=1e-6)
                traces.append(trace)
        self.assertGreater(len(set(selections)),1)
        self.assertTrue(all(trace==traces[0] for trace in traces))
        self.assertEqual(len(traces[0]),14)

    def test_official_routing_changes_do_not_skip_unselected_verification(self):
        model,spec,project=self.fixture();prefixes=[]
        for hidden in [[.4,-.2],[-.4,-.6]]:
            trace=[];attempt=spec.attempt()
            def reject(operation,values):
                trace.append(operation['id'])
                if operation['id']=='expert/0/up':raise ValueError('public verification fixture rejected')
                return project(operation,values)
            with self.assertRaises(ValueError):attempt.run(hidden,reject)
            self.assertTrue(attempt.failed);prefixes.append(trace)
        self.assertEqual(*prefixes)
        self.assertEqual(prefixes[0][-1],'expert/0/up')

    def test_complete_hyper_moe_residual_branch_matches_source_over_four_updates(self):
        from flash_hyper_ops import HyperProgram,TorchHyperConnection
        torch=self.torch;moe,spec,moe_project=self.fixture()
        config=SimpleNamespace(hc_count=3,hidden_size=2,hc_lowrank=2,rms_norm_eps=1e-6)
        reference=self.classes['Qwen4ExpTextGatedResidual'](config).eval()
        with torch.no_grad():
            for index,parameter in enumerate(reference.parameters()):
                parameter.copy_(torch.linspace(-.2,.3,parameter.numel()).reshape(parameter.shape)+(index+1)*.02)
        names={'hyper/down':'input_mix_weight_down','hyper/up':'input_mix_weight_up','hyper/inject':'block_inject_weight'}
        trace=[]
        def projection(operation,value):
            trace.append(operation['id'])
            return getattr(reference,names[operation['id']])(value)
        candidate=TorchHyperConnection(HyperProgram('a'*64,'b'*64,3,2,2,1e-6),reference.hc_norm.weight,projection)
        def local_moe(value):
            return torch.tensor(spec.attempt().run(value.reshape(-1).tolist(),moe_project)).reshape_as(value)
        for initial in [[.1,-.3,.5,-.7,.9,-.2],[-.4,.2,.7,.3,-.1,.8]]:
            state=torch.tensor(initial).reshape(1,1,6);expected=state.clone();trace.clear()
            with torch.no_grad():
                for _step in range(4):
                    mixed,residual,inject=reference(expected)
                    expected=residual+(moe(mixed).unsqueeze(-2)*inject.unsqueeze(-1)).flatten(-2)
                    state=candidate.mlp_branch(state,local_moe)
                    torch.testing.assert_close(state,expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(trace,['hyper/down','hyper/up','hyper/inject']*4)

    def test_hyper_input_shape_and_failed_projection_cannot_reach_the_moe_branch(self):
        from flash_hyper_ops import HyperProgram,TorchHyperConnection
        torch=self.torch
        def rejected(*args):raise ValueError('public rejected projection')
        candidate=TorchHyperConnection(HyperProgram('a'*64,'b'*64,3,2,2,1e-6),torch.zeros(6),rejected)
        def forbidden(*args):self.fail('unverified hyperconnection reached MoE')
        for value in [torch.zeros(1,5),torch.full((1,6),float('nan')),torch.zeros(1,6)]:
            with self.assertRaises(ValueError):candidate.mlp_branch(value,forbidden)

    def test_router_normalization_and_ties_use_official_torch_semantics(self):
        from oblivious_experts import torch_router
        torch=self.torch
        for normalize in [True,False]:
            for logits in [[0.,0.,0.],[-100.,0.,100.],[.4,.2,.9]]:
                probabilities=torch.softmax(torch.tensor(logits),dim=-1,dtype=torch.float32)
                weights,indices=probabilities.topk(2)
                if normalize:weights/=weights.sum()
                actual_ids,actual_weights=torch_router(logits,2,normalize)
                self.assertEqual(actual_ids,indices.tolist());self.assertEqual(actual_weights,weights.tolist())


if __name__=='__main__':unittest.main()
