"""Opt-in pinned official-source component comparison; public small FP32 fixtures, not Flash weights."""
import ast
import hashlib
import json
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
        record=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())['operatorReference']
        path=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['files'][path.name]['sha256']:
            raise ValueError('Pinned official Flash reference source required')
        allowed={'Qwen4ExpTextMLP','Qwen4ExpTextExperts','Qwen4ExpTextTopKRouter','Qwen4ExpTextSparseMoeBlock'}
        nodes=[node for node in ast.parse(data).body if isinstance(node,ast.ClassDef) and node.name in allowed]
        if {node.name for node in nodes}!=allowed:raise ValueError('Complete reviewed reference classes required')
        # Use only the inspected pure-PyTorch classes, without the optional fused dispatch
        # decorator or any imports/model code from the downloaded module.
        for node in nodes:node.decorator_list=[]
        scope=dict(torch=torch,nn=nn,F=F,ACT2FN={'silu':F.silu})
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
        cls.classes=scope

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
        torch=self.torch;model,spec,project=self.fixture();traces=[]
        with torch.no_grad():
            for hidden in [[.4,-.2],[-.5,.9],[1.2,.3]]:
                expected=model(torch.tensor(hidden).reshape(1,1,2)).reshape(-1)
                trace=[]
                def observed(operation,values):trace.append(dict(operation));return project(operation,values)
                actual=spec.attempt().run(hidden,observed)
                torch.testing.assert_close(torch.tensor(actual),expected,rtol=1e-5,atol=1e-6)
                traces.append(trace)
        self.assertEqual(traces[0],traces[1]);self.assertEqual(traces[1],traces[2])
        self.assertEqual(len(traces[0]),14)

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
