"""Actual Torch/quantizer checks for streaming/lowered execution; tiny public fixtures."""
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE')=='1','opt-in pinned native/Torch implementation')
class StreamedFlashMoETests(unittest.TestCase):
    def test_actual_provider_loop_reads_owned_signed_tiles_and_closes_on_failure(self):
        import sys
        from pathlib import Path
        import numpy as np
        import full_provider
        import plan_backends
        from full_plan import FullPlan,canonical
        from protocol import digest
        from public_weights import PublicTileReader
        for dtype,parameters,suffix in [('int8',{},'.i8'),('<i2',dict(weightEncoding='signed-int12-in-int16',quantizationBits=12),'.i16')]:
            for reject in [False,True]:
                with tempfile.TemporaryDirectory() as directory:
                    values=np.arange(-17,18,dtype=dtype).reshape(5,7)
                    raw=values.tobytes();identity=digest(raw)
                    (Path(directory)/(identity+suffix)).write_bytes(raw)
                    rows=[dict(name=name,weightId=identity,inputWidth=5,outputWidth=7) for name in ['first','last']]
                    plan=FullPlan(canonical(dict(modelId='a'*64,steps=2,inputTileWidth=2,outputTileWidth=3,
                                               operationTimeoutSeconds=1,parameters=parameters,projections=rows)))
                    key_data=b'public-evaluation-key';key_id=digest(key_data);session='s'*32
                    frames=[(plan.frame(session,key_id,0,'evaluation-key'),key_data,0)]
                    ordered=list(plan.schedule())*2
                    frames += [(plan.frame(session,key_id,index+1,'evaluate',tile['id']),b'ciphertext',0) for index,tile in enumerate(ordered)]
                    frames.append((plan.frame(session,key_id,len(ordered)+1,'finish'),b'',0))
                    observed=[];readers=[];sent=[]
                    class Reader(PublicTileReader):
                        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);readers.append(self)
                    def evaluate(encrypted,matrix,key):
                        tile=ordered[len(observed)];observed.append(matrix.copy())
                        expected=values[tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']].astype(np.int64).astype(np.uint64)
                        np.testing.assert_array_equal(matrix,expected);self.assertTrue(matrix.flags.c_contiguous)
                        if reject:raise RuntimeError('native evaluation failed')
                        return SimpleNamespace(serialize=lambda:b'encrypted-result')
                    args=SimpleNamespace(directory=directory,plan_id=plan.identity,profile='fixture-only',ssh_route=None,public_source_dir=None)
                    stream=SimpleNamespace(buffer=SimpleNamespace(raw=object()))
                    with patch.object(sys,'argv',['provider']),patch.object(sys,'stdin',stream),patch.object(sys,'stdout',stream), \
                         patch.object(plan_backends,'provider_arguments',return_value=args),patch.object(plan_backends,'load_backend_plan',return_value=plan), \
                         patch.object(full_provider,'matrix_evaluator',return_value=evaluate),patch.object(full_provider,'prepare_evaluator_key'), \
                         patch.object(full_provider,'ext',SimpleNamespace(CpuCompressionKey=SimpleNamespace(deserialize=lambda data:object()),EncryptedMatrix=SimpleNamespace(deserialize=lambda data:object()))), \
                         patch.object(full_provider,'receive',side_effect=frames),patch.object(full_provider,'send',side_effect=lambda *args,**kwargs:sent.append(args)), \
                         patch.object(full_provider,'PublicTileReader',Reader):
                        if reject:
                            with self.assertRaises(RuntimeError):full_provider.serve()
                        else:full_provider.serve()
                    self.assertEqual(len(observed),1 if reject else len(ordered))
                    self.assertEqual(len(sent),1 if reject else len(ordered)+1)
                    self.assertTrue(readers[0].closed);self.assertIsNone(readers[0].handle)

    def test_owned_staged_whole_object_materialization_survives_source_cleanup(self):
        import torch
        from pathlib import Path
        from flash_stage_source import staged_source
        from test_flash_staging import fixture
        name,spec,files=fixture()
        with tempfile.TemporaryDirectory() as parent:
            def acquire(directory,unused,**kwargs):
                for file,data in files.items():(Path(directory)/file).write_bytes(data)
            with patch('flash_stage_source.staging_spec',return_value=spec),patch('flash_stage_source.provision_spec',side_effect=acquire):
                with staged_source(parent,'decoder',0) as source:owned=source.tensor(name)
            self.assertTrue(source.closed);self.assertEqual(list(Path(parent).iterdir()),[])
            self.assertEqual(owned.dtype,torch.float32);torch.testing.assert_close(owned,torch.tensor([[1.,2.],[3.,4.]]),rtol=0,atol=0)
            with self.assertRaises(ValueError):source.tensor(name)

    def test_full_qwen_reference_rejection_cannot_truncate_the_provider_route(self):
        import torch
        import full_model
        from full_status import ExecutionRejected
        from verified_execution import VALIDATION_BOUNDARY
        transcripts=[]
        for reject_step in [None,0,2,3]:
            trace=[];wire=[];bridge=None
            plan=SimpleNamespace(identity='a'*64,protocol_id='test-only',definition=lambda:dict(modelId='b'*64))
            class Bridge:
                def __init__(self,*args,**kwargs):
                    self.plan=plan;self.sequence=1;self.closed=False;self.replay_seconds=self.remote_seconds=0
                    self.sent=self.received=self.key_bytes=0
                def finish(self):
                    self.assert_complete();wire.append('finish')
                def assert_complete(self):
                    if self.sequence!=1549:raise AssertionError('incomplete route')
                def close(self):
                    if not self.closed:wire.append('close');self.closed=True
            def spawn(*args,**kwargs):
                nonlocal bridge
                bridge=Bridge();return bridge
            def step(model,projections,cache,tokens,original_count,index,mode,bucket,pad):
                trace.append((mode,index));self.assertEqual(tokens,[7,8]+[0]*index)
                if mode=='execute':
                    self.assertFalse(bridge.closed);bridge.sequence+=387;wire.append(('execute',index))
                else:self.assertTrue(bridge.closed,'reference comparison ran with live provider transport')
                chosen=1 if mode=='clear-reference' and index==reject_step else 0
                return torch.tensor([2.,1.] if chosen==0 else [1.,2.]),.1
            projection=SimpleNamespace(bridge=None,secret=None,parameters=None)
            settings=dict(cpuThreads=2,steps=4,contextBucket=8,backendId='test-only')
            prepared=(None,SimpleNamespace(encode=lambda text:[7,8],decode=lambda tokens:'public-fixture'),
                      SimpleNamespace(glwe_crypto_params=None),[projection],plan)
            with patch.object(full_model,'qwen_backend',return_value=(settings,None,lambda:{},None)), \
                 patch.object(full_model,'prepare',return_value=prepared),patch.object(full_model,'FullBridge',side_effect=spawn), \
                 patch.object(full_model,'model_step',side_effect=step),patch.object(full_model.ext,'cpu_create_private_key',return_value=(object(),object())), \
                 patch.dict(os.environ,{'RAYON_NUM_THREADS':'2'}):
                if reject_step is None:
                    report=full_model.run('unused','unused','public-fixture')
                    self.assertEqual(report['validationBoundary'],VALIDATION_BOUNDARY)
                    self.assertEqual(report['clientLocalOutputTokens'],[0,0,0,0])
                else:
                    with self.assertRaises(ExecutionRejected):full_model.run('unused','unused','public-fixture')
            self.assertEqual(trace[:4],[('execute',i) for i in range(4)])
            self.assertIsNone(projection.secret);self.assertIsNone(projection.bridge)
            transcripts.append(wire)
        for transcript in transcripts[1:]:self.assertEqual(transcript,transcripts[0])

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
            with actual.coefficient_matrix() as weights:
                self.assertEqual(weights.dtype,np.dtype('<i2'));self.assertGreater(int(abs(weights).max()),127)
            value=torch.tensor([[.1,-.2,.3,-.4]],dtype=torch.float32);actual.mode=control.mode='clear-reference'
            torch.testing.assert_close(actual(value),control(value),rtol=0,atol=0)

    def test_full_linear_tiled_dispatch_uses_owned_bridge_geometry(self):
        import numpy as np
        import torch
        import full_model
        from full_plan import FullPlan,canonical
        from full_bridge import FullBridge
        from full_admission import FullRouteAdmission
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        with tempfile.TemporaryDirectory() as directory:
            layer=torch.nn.Linear(5,3,bias=True).eval().requires_grad_(False)
            with torch.no_grad():
                layer.weight.copy_(torch.arange(15,dtype=torch.float32).reshape(3,5)*.07-.4)
                layer.bias.copy_(torch.tensor([.1,-.1,.3]))
            projection=full_model.FullLinear('public',layer,GLWELinearLayerExecutor(use_dynamic_quantization=True),directory,12)
            value=torch.tensor([[.1,-.2,.3,-.4,.5]]);projection.mode='clear-reference';expected=projection(value)
            bridge=FullBridge.__new__(FullBridge)
            bridge.plan=FullPlan(canonical(dict(steps=1,contextBucket=1,inputTileWidth=2,outputTileWidth=2,
                parameters=dict(parameters=dict(bits_reserved_for_computation=35)),projections=[projection.definition])))
            bridge.admission=FullRouteAdmission(bridge.plan);bridge.cancelled=None;bridge.sequence=1
            def evaluate(encrypted,matrix,tile):
                bridge.admission.admit_tile(tile,matrix.shape,bridge.sequence)
                result=encrypted.astype(np.int64)@matrix.view(np.int64)
                bridge.admission.verified();bridge.sequence+=1;return result
            bridge.evaluate=evaluate
            projection.release_float_reference();projection=projection.prepared_copy()
            projection.bridge=bridge;projection.secret=object();projection.parameters=object();projection.mode='execute'
            # Exercise actual Torch/NumPy quantization, route admission, tile
            # accumulation and dequantization; crypto is explicitly simulated.
            native=SimpleNamespace(encrypt_matrix=lambda **kwargs:kwargs['data'].copy(),decrypt_matrix=lambda result,*args:result)
            with patch.object(full_model,'ext',native):actual=projection(value)
            bridge.admission.complete();self.assertEqual(bridge.sequence,7)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_prepared_projection_releases_float_weights_and_isolates_attempt_state(self):
        import gc
        import weakref
        import torch
        from full_model import FullLinear
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        for bits in [8,12]:
            with tempfile.TemporaryDirectory() as directory:
                layer=torch.nn.Linear(5,3,bias=True).eval().requires_grad_(False)
                projection=FullLinear('prepared',layer,GLWELinearLayerExecutor(use_dynamic_quantization=True),directory,bits)
                value=torch.tensor([[.1,-.2,.3,-.4,.5]])
                projection.mode='clear-reference';expected=projection(value)
                weight=weakref.ref(layer.weight);del layer
                with self.assertRaises(ValueError):projection.prepared_copy()
                projection.release_float_reference();projection.release_float_reference();gc.collect()
                self.assertIsNone(weight());self.assertIsNone(projection.remote)
                first=projection.prepared_copy();second=projection.prepared_copy()
                torch.testing.assert_close(first(value),expected,rtol=0,atol=0)
                torch.testing.assert_close(second(value),expected,rtol=0,atol=0)
                self.assertNotEqual(first.scale.data_ptr(),second.scale.data_ptr())
                first.scale.zero_();self.assertFalse(torch.equal(first.scale,second.scale))
                first.definition['weightId']='changed';self.assertNotEqual(first.definition,second.definition)
                first.context.input_quantizers[0].quant_options.n_bits=1
                self.assertEqual(second.context.input_quantizers[0].quant_options.n_bits,bits)
                first.bridge=object();first.secret=object();first.parameters=object()
                self.assertIsNone(second.bridge);self.assertIsNone(second.secret);self.assertIsNone(projection.secret)
                second.mode='float-reference'
                with self.assertRaisesRegex(ValueError,'explicitly released'):second(value)
                second.mode='clear-reference';path=first.weight_path;original=path.read_bytes()
                changed=bytes([original[0]^1])+original[1:];path.write_bytes(changed)
                with self.assertRaisesRegex(ValueError,'coefficient identity changed'):second(value)
                path.write_bytes(original)
                torch.testing.assert_close(second(value),expected,rtol=0,atol=0)
                def cancel():raise InterruptedError('cancelled before prepared coefficients used')
                second.bridge=SimpleNamespace(check_active=cancel)
                with self.assertRaises(InterruptedError):second(value)
                second.bridge=None
                path.unlink()
                with self.assertRaises(FileNotFoundError):second(value)

    def test_public_matrix_scope_closes_on_success_failure_and_cache_changes(self):
        from pathlib import Path
        import numpy as np
        from public_weights import mapped_public_matrix
        for dtype in ['int8','<i2']:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'public';expected=np.arange(15,dtype=dtype).reshape(5,3);path.write_bytes(expected.tobytes())
                for fail in [False,True]:
                    matrix=None
                    try:
                        with mapped_public_matrix(path,(5,3),dtype) as matrix:
                            self.assertFalse(matrix.flags.writeable);owned=np.array(matrix,copy=True)
                            alias=path.with_suffix('.owned-link');os.link(path,alias);alias.unlink()
                            if fail:raise RuntimeError('cancelled computation')
                    except RuntimeError:
                        if not fail:raise
                    self.assertTrue(matrix._mmap.closed);np.testing.assert_array_equal(owned,expected)
                with patch.object(np,'memmap',side_effect=AssertionError('mapping before geometry check')):
                    with self.assertRaises(ValueError):
                        with mapped_public_matrix(path,(6,3),dtype):self.fail('bad geometry accepted')
                matrix=None
                with self.assertRaises(ValueError):
                    with mapped_public_matrix(path,(5,3),dtype) as matrix:
                        path.write_bytes((expected+1).tobytes())
                self.assertTrue(matrix._mmap.closed)

    def test_full_linear_cache_opens_lazily_and_closes_on_encrypted_failure(self):
        import numpy as np
        import torch
        import full_model
        from full_bridge import FullBridge
        from full_admission import FullRouteAdmission
        from full_plan import FullPlan,canonical
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        for bits,message_bits in [(8,27),(12,35)]:
            with tempfile.TemporaryDirectory() as directory:
                mappings=[];open_matrix=np.memmap
                def record(*args,**kwargs):
                    result=open_matrix(*args,**kwargs);mappings.append(result._mmap);return result
                with patch.object(np,'memmap',side_effect=record):
                    layer=torch.nn.Linear(5,3).eval().requires_grad_(False)
                    projection=full_model.FullLinear('public',layer,GLWELinearLayerExecutor(use_dynamic_quantization=True),directory,bits)
                    self.assertEqual(mappings,[])
                    value=torch.tensor([[.1,-.2,.3,-.4,.5]]);projection.mode='clear-reference';projection(value)
                    self.assertEqual(len(mappings),1);self.assertTrue(mappings[-1].closed)
                    bridge=FullBridge.__new__(FullBridge);bridge.cancelled=None
                    bridge.plan=FullPlan(canonical(dict(steps=1,contextBucket=1,inputTileWidth=2,outputTileWidth=2,
                        parameters=dict(parameters=dict(bits_reserved_for_computation=message_bits)),projections=[projection.definition])))
                    bridge.admission=FullRouteAdmission(bridge.plan)
                    def reject(*args):raise InterruptedError('cancelled encrypted tile')
                    bridge.evaluate=reject;projection.bridge=bridge;projection.secret=object();projection.parameters=object();projection.mode='execute'
                    with patch.object(full_model,'ext',SimpleNamespace(encrypt_matrix=lambda **kwargs:kwargs['data'])):
                        with self.assertRaises(InterruptedError):projection(value)
                    self.assertEqual(len(mappings),2);self.assertTrue(all(mapping.closed for mapping in mappings))
                    projection.weight_path.unlink();projection.mode='clear-reference'
                    with patch.object(layer,'forward',side_effect=AssertionError('cache miss silently fell back')):
                        with self.assertRaises(FileNotFoundError):projection(value)
                    self.assertEqual(len(mappings),2)

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
            with reference.coefficient_matrix() as weights:self.assertGreater(int(abs(weights).max()),127)
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
        from contextlib import nullcontext
        projection.context=None;projection.coefficient_matrix=lambda:nullcontext(np.full((width,1),127,dtype=np.int8))
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
            definitions=model.prepare_projections(flash_moe.lowered_operations(program))
            plan=SimpleNamespace(definition=lambda:dict(projections=definitions))
            for hidden in [[.4,-.2],[-.4,-.6]]:
                reads.clear()
                actual=model.evaluate(hidden,'float-reference',plan)
                expected=torch.tensor(program.attempt().run(hidden,direct))
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                self.assertEqual(len(reads),11)
                self.assertEqual([expert for name,expert in reads if expert is not None],[0,0,1,1,2,2])
                reads.clear();quantized=model.evaluate(hidden,'clear-reference',plan)
                self.assertEqual(reads,[]);self.assertTrue(all(projection.remote is None for projection in model.prepared.values()))
                self.assertTrue(torch.isfinite(quantized).all())
                fresh=model.projection
                with patch.object(model,'projection',side_effect=lambda name,mode:fresh(name)):
                    rebuilt=model.evaluate(hidden,'clear-reference',plan)
                torch.testing.assert_close(quantized,rebuilt,rtol=0,atol=0)
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
            def observed(name,mode='float-reference'):trace.append(name);return original_projection(name,mode)
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
