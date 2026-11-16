"""Actual Torch/quantizer checks for streaming/lowered execution; tiny public fixtures."""
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE')=='1','opt-in pinned native/Torch implementation')
class StreamedFlashMoETests(unittest.TestCase):
    def test_unavailable_requested_native_moe_rejects_before_source_or_key_creation(self):
        import flash_moe
        import concrete_ml_extensions as ext
        def unavailable():raise ValueError('native implementation unavailable')
        with patch.object(flash_moe,'moe_backend',return_value=({},None,unavailable)),              patch.object(flash_moe,'FlashSource') as source,patch.object(ext,'cpu_create_private_key') as keys:
            with self.assertRaises(ValueError):flash_moe.StreamedMoE('/unused','/unused','flash-dense-moe-ntt')
            source.assert_not_called();keys.assert_not_called()

    def test_last_layer_expert_materialization_requires_slicing_and_owns_the_public_tensor(self):
        from pathlib import Path
        import torch
        from safetensors.torch import save_file
        import flash_weights
        name='model.language_model.layers.47.mlp.experts.gate_up_proj'
        expected=torch.arange(24,dtype=torch.float32).reshape(2,3,4).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'public.safetensors';save_file({name:expected},str(path))
            source=flash_weights.FlashSource.__new__(flash_weights.FlashSource);source.directory=Path(directory)
            source.spec=dict(tensors={name:dict(shard=path.name,shape=[2,3,4],dtype='BF16')})
            source.signatures={path.name:flash_weights.file_signature(path)}
            with self.assertRaises(ValueError):source.tensor(name)
            with self.assertRaises(ValueError):source.tensor(name,expert=2)
            actual=source.tensor(name,expert=1);path.unlink()
            self.assertEqual(actual.dtype,torch.float32);torch.testing.assert_close(actual,expected[1].float(),rtol=0,atol=0)

    def test_full_linear_uses_wide_cache_and_matches_the_clear_twelve_bit_reference(self):
        import numpy as np
        import torch
        from torch import nn
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from full_model import FullLinear
        from wide_reference import WideReferenceLinear
        with tempfile.TemporaryDirectory() as directory:
            layer=nn.Linear(4,3,bias=True).eval().requires_grad_(False)
            with torch.no_grad():
                layer.weight.copy_(torch.arange(12,dtype=torch.float32).reshape(3,4)*.07-.4);layer.bias.zero_()
            executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
            actual=FullLinear('public',layer,executor,directory,12)
            control=WideReferenceLinear(SimpleNamespace(name='public',remote=layer),executor,directory,12)
            self.assertEqual(actual.weights.dtype,np.dtype('<i2'));self.assertGreater(int(abs(actual.weights).max()),127)
            value=torch.tensor([[.1,-.2,.3,-.4]],dtype=torch.float32);actual.mode=control.mode='clear-reference'
            torch.testing.assert_close(actual(value),control(value),rtol=0,atol=0)

    def test_bounded_integer_reference_matches_full_int64_product_without_float_rounding(self):
        import numpy as np
        import torch
        from integer_reference import integer_dot,POLICY
        values=torch.arange(2*2050,dtype=torch.int64).reshape(2,2050)%251
        weights=(np.arange(2050*2051,dtype=np.int64).reshape(2050,2051)%255-128).astype(np.int16)
        expected=torch.matmul(values,torch.from_numpy(weights.astype(np.int64)))
        sizes=[];original=np.array
        def bounded(value,*args,**kwargs):sizes.append(value.shape);return original(value,*args,**kwargs)
        with patch('integer_reference.np.array',side_effect=bounded):actual=integer_dot(values,weights)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(all(shape[0]<=POLICY['inputTileWidth'] and shape[1]<=POLICY['outputTileWidth'] for shape in sizes))

    def test_wide_reference_preserves_coefficients_and_cannot_enter_encrypted_mode(self):
        import torch
        from torch import nn
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from wide_reference import WideReferenceLinear
        with tempfile.TemporaryDirectory() as directory:
            layer=nn.Linear(4,3,bias=False).eval().requires_grad_(False)
            with torch.no_grad():layer.weight.copy_(torch.arange(12,dtype=torch.float32).reshape(3,4)*.07-.4)
            reference=WideReferenceLinear(SimpleNamespace(name='public',remote=layer),GLWELinearLayerExecutor(use_dynamic_quantization=True),directory,12)
            self.assertGreater(int(abs(reference.weights).max()),127)
            value=torch.tensor([[.1,-.2,.3,-.4]],dtype=torch.float32);reference.mode='clear-reference'
            torch.testing.assert_close(reference(value),layer(value),rtol=.01,atol=.001)
            reference.mode='execute'
            with self.assertRaises(ValueError):reference(value)

    def test_model_step_preserves_bucket_and_rejects_cache_history_mismatch(self):
        import torch
        from full_model import model_step
        class Cache:
            length=0
            def get_seq_length(self):return self.length
            def crop(self,length):self.length=length
        cache=Cache();calls=[];projection=SimpleNamespace(mode=None)
        def model(**arguments):
            values=arguments['input_ids'];calls.append(arguments)
            cache.length+=values.shape[1]
            return SimpleNamespace(logits=torch.arange(values.shape[1]*4,dtype=torch.float32).reshape(1,values.shape[1],4))
        logits,_=model_step(model,[projection],cache,[1,2,3],3,0,'clear-reference',8,151643)
        self.assertEqual(logits.tolist(),[8,9,10,11]);self.assertEqual(cache.length,3)
        self.assertEqual(calls[0]['input_ids'].tolist(),[[1,2,3]+[151643]*5])
        self.assertEqual(calls[0]['attention_mask'].tolist(),[[1,1,1,0,0,0,0,0]])
        model_step(model,[projection],cache,[1,2,3,4],3,1,'clear-reference',8,151643)
        self.assertEqual(calls[1]['attention_mask'].shape,(1,4));cache.length=5
        with self.assertRaises(ValueError):model_step(model,[projection],cache,[1,2,3,4,5],3,2,'execute',8,151643)
        self.assertEqual(len(calls),2)

    def test_integer_source_buffers_keep_exact_int64_values(self):
        from pathlib import Path
        import torch
        import flash_weights
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'public.safetensors';name='public.hash_multipliers'
            expected=torch.tensor([2**60+3,-(2**59)+7],dtype=torch.int64)
            save_file({name:expected},str(path));source=flash_weights.FlashSource.__new__(flash_weights.FlashSource)
            source.directory=Path(directory);source.signatures={path.name:flash_weights.file_signature(path)}
            source.spec=dict(tensors={name:dict(shard=path.name,shape=[2],dtype='I64')})
            actual=source.tensor(name)
            self.assertEqual(actual.dtype,torch.int64);self.assertEqual(actual.tolist(),expected.tolist())
            path.unlink();self.assertEqual(actual.tolist(),expected.tolist())

    def test_missing_complete_ple_table_rejects_before_any_private_key_creation(self):
        import concrete_ml_extensions as ext
        import flash_ple_model
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(flash_ple_model,'versions',return_value={}), \
                 patch.dict(os.environ,{'RAYON_NUM_THREADS':'4'}), \
                 patch.object(ext,'cpu_create_private_key',side_effect=AssertionError('missing table generated a key')) as keys:
                with self.assertRaises(FileNotFoundError):flash_ple_model.run(directory,directory,'unused-public-reference',local_storage_limit_bytes=110000000000)
                keys.assert_not_called()

    def test_full_model_source_mutation_during_loader_is_rejected(self):
        import hashlib
        import json
        from pathlib import Path
        import full_model
        for rewrite in [False,True]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);weights=root/'public.bin';weights.write_bytes(b'original')
                metadata=root/'source.json';metadata.write_text(json.dumps(dict(source=dict(files={'public.bin':hashlib.sha256(b'original').hexdigest()}))))
                def load(*args,**kwargs):
                    if rewrite:weights.write_bytes(b'changed!')
                    return SimpleNamespace(eval=lambda:None)
                settings=dict(cpuThreads=1)
                with patch.object(full_model,'qwen_backend',return_value=(settings,None,lambda:{},None)), \
                     patch.object(full_model,'Path',return_value=SimpleNamespace(with_name=lambda name:metadata)), \
                     patch.object(full_model.AutoModelForCausalLM,'from_pretrained',side_effect=load), \
                     patch.object(full_model.AutoTokenizer,'from_pretrained',side_effect=RuntimeError('public stop after materialization')):
                    with self.assertRaises(ValueError if rewrite else RuntimeError):full_model.prepare(directory,root/'cache')

    def test_clear_quantized_reference_accumulates_exact_integers_above_float32_precision(self):
        import numpy as np
        import torch
        from full_model import FullLinear
        width=4098;values=torch.full((1,width),255,dtype=torch.float32);values[0,0]=0
        projection=FullLinear.__new__(FullLinear);torch.nn.Module.__init__(projection)
        projection.mode='clear-reference';projection.definition=dict(inputWidth=width,outputWidth=1)
        projection.context=None;projection.weights=np.full((width,1),127,dtype=np.int8)
        projection.scale=projection.zero_point=projection.sum_weights=None;projection.remote=SimpleNamespace(bias=None)
        projection.executor=SimpleNamespace(_dynamic_input_quantization=lambda *args:(values,1,0,values.shape),
                                            _apply_correction_and_dequantize=lambda raw,*args:raw)
        result=projection(values)
        expected=(width-1)*255*127
        self.assertEqual(result.dtype,torch.int64);self.assertEqual(int(result[0,0]),expected)
        self.assertNotEqual(int(torch.tensor(expected,dtype=torch.float32)),expected)

    def test_client_convolution_tensor_is_whole_and_expert_banks_require_slicing(self):
        from pathlib import Path
        import torch
        import flash_weights
        from safetensors.torch import save_file
        for name,shape,expert_bank in [('model.language_model.layers.0.linear_attn.conv1d.weight',(4,1,3),False),
                                       ('model.language_model.layers.0.mlp.experts.gate_up_proj',(2,4,3),True)]:
            with tempfile.TemporaryDirectory() as directory:
                shard='tiny.safetensors';path=Path(directory)/shard;expected=torch.ones(shape,dtype=torch.bfloat16)
                save_file({name:expected},str(path));source=flash_weights.FlashSource.__new__(flash_weights.FlashSource)
                source.directory=Path(directory);source.signatures={shard:flash_weights.file_signature(path)}
                source.spec=dict(tensors={name:dict(shard=shard,shape=list(shape),dtype='BF16')})
                if expert_bank:
                    with self.assertRaises(ValueError):source.tensor(name)
                    torch.testing.assert_close(source.tensor(name,expert=0),expected[0].float())
                else:
                    torch.testing.assert_close(source.tensor(name),expected.float())
                    with self.assertRaises(ValueError):source.tensor(name,expert=0)

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
            source.spec=spec
            with patch.object(flash_weights,'file_signature',side_effect=[signature,('changed',)]):
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
            # Compose the actual streaming adapter with normalization and three real
            # quantized hyperconnection modules, under one complete ordered route.
            import flash_mlp
            from flash_hyper import FlashHyperModel
            from flash_hyper_ops import HyperProgram,TorchHyperConnection
            from full_model import FullLinear
            hyper=FlashHyperModel.__new__(FlashHyperModel);hyper.executor=model.executor;hyper.projections=[]
            hyper_program=HyperProgram('a'*64,'b'*64,3,2,2,1e-6)
            for operation in hyper_program.contract()['operations']:
                linear=torch.nn.Linear(operation['inputWidth'],operation['outputWidth'],bias=False)
                with torch.no_grad():linear.weight.copy_(torch.linspace(-.2,.3,linear.weight.numel()).reshape_as(linear.weight))
                hyper.projections.append(FullLinear(operation['id'],linear,model.executor,directory))
            by_name={projection.name:projection for projection in hyper.projections}
            hyper.graph=TorchHyperConnection(hyper_program,torch.zeros(6),lambda operation,value:by_name[operation['id']](value))
            complete=flash_mlp.StreamedFlashMLP.__new__(flash_mlp.StreamedFlashMLP);complete.moe=model;complete.hyper=hyper
            prefix=[projection.definition for projection in hyper.projections]
            all_definitions=prefix+definitions
            complete_plan=SimpleNamespace(definition=lambda:dict(projections=all_definitions))
            trace=[];original_projection=model.projection
            def observed(name):trace.append(name);return original_projection(name)
            model.projection=observed
            handles=[projection.register_forward_pre_hook(lambda module,args:trace.append(module.name)) for projection in hyper.projections]
            try:
                value=torch.tensor([[.1,-.3,.5,-.7,.9,-.2]])
                with torch.no_grad():
                    mixed,residual,inject=hyper.graph(value)
                    expected=hyper.graph.combine(residual,inject,torch.tensor(program.attempt().run(mixed.reshape(-1).tolist(),direct)).reshape_as(mixed))
                trace.clear()
                actual=complete.evaluate(value,'float-reference',complete_plan)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                self.assertEqual(trace,[row['name'] for row in all_definitions])
                trace.clear();self.assertTrue(torch.isfinite(complete.evaluate(value,'clear-reference',complete_plan)).all())
                self.assertEqual(trace,[row['name'] for row in all_definitions])
                corrupt=[dict(row) for row in all_definitions];corrupt[0]['weightId']='c'*64;trace.clear()
                with self.assertRaises(ValueError):complete.evaluate(value,'clear-reference',SimpleNamespace(definition=lambda:dict(projections=corrupt)))
                self.assertEqual(trace,[])
            finally:
                for handle in handles:handle.remove()
                model.projection=original_projection
            changed=[dict(row) for row in definitions];changed[0]['weightId']='b'*64
            with self.assertRaises(ValueError):model.evaluate([.4,-.2],'clear-reference',SimpleNamespace(definition=lambda:dict(projections=changed)))


if __name__=='__main__':unittest.main()
