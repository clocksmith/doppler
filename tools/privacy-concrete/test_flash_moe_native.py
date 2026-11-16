"""Actual Torch/quantizer checks for streaming/lowered execution; tiny public fixtures."""
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE')=='1','opt-in pinned native/Torch implementation')
class StreamedFlashMoETests(unittest.TestCase):
    def test_source_rewrite_during_tensor_copy_is_rejected(self):
        from pathlib import Path
        import torch
        import flash_weights
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            name='public.weight';shard='test.safetensors';path=Path(directory)/shard
            save_file({name:torch.ones((2,2),dtype=torch.bfloat16)},str(path))
            signature=flash_weights.file_signature(path)
            source=flash_weights.FlashSource.__new__(flash_weights.FlashSource)
            source.directory=Path(directory);source.signatures={shard:signature}
            spec=dict(tensors={name:dict(shard=shard,shape=[2,2],dtype='BF16')})
            with patch.object(flash_weights,'SPEC',spec),patch.object(flash_weights,'file_signature',side_effect=[signature,('changed',)]):
                with self.assertRaises(ValueError):source.tensor(name)

    def test_combined_gate_up_lowering_matches_separate_complete_operator_reference(self):
        import torch
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        import flash_moe
        from oblivious_experts import ExpertProgram,FlashMoEProgram
        program=FlashMoEProgram(ExpertProgram('a'*64,3,2,2,2),2,True,'b'*64)
        matrices={}
        for index,row in enumerate(flash_moe.lowered_operations(program)):
            matrices[row['name']]=torch.linspace(-.4,.6,row['inputWidth']*row['outputWidth']).reshape(row['outputWidth'],row['inputWidth'])+(index+1)*.03
        reads=[]
        def source_tensor(name,expert=None):
            reads.append((name,expert))
            leaf=name.rsplit('.',1)[-1]
            if expert is not None:key=f'expert/{expert}/'+leaf[:-5]
            elif name.endswith('shared_expert_gate.weight'):key='shared/mix'
            elif name.endswith('.gate.weight'):key='router/logits'
            else:key='shared/'+name.split('.')[-2][:-5]
            return matrices[key]
        def direct(operation,values):
            name=operation['id'];value=torch.tensor(values,dtype=torch.float32)
            if name.startswith('expert/') and name.endswith(('/gate','/up')):
                key=name.rsplit('/',1)[0]+'/gate_up';matrix=matrices[key]
                matrix=matrix[:2] if name.endswith('/gate') else matrix[2:]
            else:matrix=matrices[name]
            return torch.nn.functional.linear(value,matrix).tolist()
        with tempfile.TemporaryDirectory() as directory,patch.object(flash_moe,'PROGRAM',program):
            model=flash_moe.StreamedMoE.__new__(flash_moe.StreamedMoE)
            model.source=SimpleNamespace(tensor=source_tensor);model.directory=directory
            model.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
            definitions=[model.projection(row['name']).definition for row in flash_moe.lowered_operations(program)]
            plan=SimpleNamespace(definition=lambda:dict(projections=definitions))
            for hidden in [[.4,-.2],[-.4,-.6]]:
                reads.clear()
                actual=model.evaluate(hidden,'float-reference',plan)
                expected=torch.tensor(program.attempt().run(hidden,direct))
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                self.assertEqual(len(reads),11)
                self.assertEqual([expert for name,expert in reads if expert is not None],[0,0,1,1,2,2])
                quantized=model.evaluate(hidden,'clear-reference',plan)
                self.assertTrue(torch.isfinite(quantized).all())
            changed=[dict(row) for row in definitions];changed[0]['weightId']='b'*64
            with self.assertRaises(ValueError):model.evaluate([.4,-.2],'clear-reference',SimpleNamespace(definition=lambda:dict(projections=changed)))


if __name__=='__main__':unittest.main()
