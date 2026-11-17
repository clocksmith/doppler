"""Real BF16 source slicing and installed per-channel quantization; no crypto."""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch quantizer')
class ChunkedPreparationTests(unittest.TestCase):
    def test_installed_quantizer_preserves_bytes_metadata_and_clear_output(self):
        import torch
        from full_model import FullLinear
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        torch.set_num_threads(2)
        generator=torch.Generator().manual_seed(82)
        for bits in (8,12):
            for outputs,chunk in ((1,1),(19,1),(19,7),(19,32)):
                with self.subTest(bits=bits,outputs=outputs,chunk=chunk),tempfile.TemporaryDirectory() as directory:
                    source=torch.nn.Linear(257,outputs,bias=True)
                    with torch.no_grad():
                        source.weight.copy_(torch.randn(source.weight.shape,generator=generator)*.3)
                        source.weight[0].fill_(0)
                        if outputs>1:source.weight[1].fill_(.125)
                    executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
                    whole=FullLinear('projection',source,executor,directory,bits)
                    actual=FullLinear('projection',source,executor,directory,bits,output_chunk_rows=chunk)
                    self.assertEqual(actual.definition,whole.definition)
                    self.assertEqual(actual.weight_path.read_bytes(),whole.weight_path.read_bytes())
                    for key in ('scale','zero_point','sum_weights'):
                        torch.testing.assert_close(getattr(actual,key),getattr(whole,key),rtol=0,atol=0)
                    value=torch.randn((1,2,257),generator=generator)
                    whole.mode=actual.mode='clear-reference'
                    torch.testing.assert_close(actual(value),whole(value),rtol=0,atol=0)

    def test_source_rows_are_sliced_before_conversion_and_compose_expert_rows(self):
        import torch
        from full_model import FullLinear
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from test_flash_streamed_moe import FlashStreamedMoETests
        with tempfile.TemporaryDirectory() as source_dir,tempfile.TemporaryDirectory() as cache:
            _,moe,_,source,_=FlashStreamedMoETests().fixture(source_dir)
            executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
            for name in ('shared_gate','expert_1_up','expert_1_down'):
                op=moe.ops[name]
                whole=FullLinear(name,op,executor,cache,12)
                with patch.object(source,'tensor',wraps=source.tensor) as read:
                    actual=FullLinear(name,op,executor,cache,12,output_chunk_rows=1)
                    offset=0 if op._rows is None else op._rows[0]
                    self.assertEqual([call.kwargs['rows'] for call in read.call_args_list],
                        [(offset+i,offset+i+1) for i in range(op.out_features)])
                self.assertEqual(actual.definition,whole.definition)
                import safetensors
                from contextlib import contextmanager
                open_source=safetensors.safe_open
                @contextmanager
                def only_slices(*args,**kwargs):
                    with open_source(*args,**kwargs) as handle:
                        class SlicesOnly:
                            def get_slice(self,name):return handle.get_slice(name)
                            def get_tensor(self,name):raise AssertionError('whole BF16 tensor materialized')
                        yield SlicesOnly()
                with patch.object(safetensors,'safe_open',side_effect=only_slices):
                    with op.materialized_weight(output_rows=(0,1)) as first:
                        self.assertEqual(tuple(first.shape),(1,op.in_features));self.assertEqual(first.dtype,torch.float32)
                with self.assertRaises(ValueError):
                    with op.materialized_weight(output_rows=(0,op.out_features+1)):self.fail('invalid rows borrowed')

    def test_cancel_during_quantization_never_publishes_partial_coefficients(self):
        import torch
        from full_model import FullLinear
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        event=threading.Event();executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
        quantize=executor._per_channel_weight_quantization
        def cancel(*args):
            result=quantize(*args);event.set();return result
        with tempfile.TemporaryDirectory() as cache,patch.object(executor,'_per_channel_weight_quantization',side_effect=cancel):
            sibling=Path(cache)/'unrelated';sibling.write_text('keep')
            with self.assertRaises(InterruptedError):
                FullLinear('projection',torch.nn.Linear(17,7),executor,cache,12,cancelled=event,output_chunk_rows=2)
            self.assertEqual(list(Path(cache).iterdir()),[sibling])


if __name__=='__main__':unittest.main()
