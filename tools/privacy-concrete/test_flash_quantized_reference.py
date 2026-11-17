"""Real quantizer and complete tiny graph; no native encryption or Flash weights."""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in real Torch quantizer')
class BoundedReferenceTests(unittest.TestCase):
    def test_cancelled_integer_reference_stops_before_later_weight_tiles(self):
        import numpy as np
        import torch
        from integer_reference import integer_dot,POLICY
        event=threading.Event();width=POLICY['inputTileWidth']+1;height=POLICY['outputTileWidth']+1
        values=torch.ones((1,width),dtype=torch.int64);weights=np.ones((width,height),dtype=np.int16)
        multiply=torch.matmul;calls=[]
        def cancel(*args):
            calls.append(True);result=multiply(*args);event.set();return result
        with patch('integer_reference.torch.matmul',side_effect=cancel):
            with self.assertRaises(InterruptedError):integer_dot(values,weights,cancelled=event)
        self.assertEqual(len(calls),1)

    def test_complete_five_row_prefill_and_decode_match_whole_quantizer_and_inventory(self):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_quantized_reference import BoundedQuantizedReference
        from full_model import FullLinear
        from full_plan import canonical
        from protocol import digest
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        torch,graph,_,context=FlashModelReferenceTests().fixture()
        whole_executor=GLWELinearLayerExecutor(use_dynamic_quantization=True);identities={}
        with ExitStack() as stack:
            whole=Path(stack.enter_context(tempfile.TemporaryDirectory()));scratch=Path(stack.enter_context(tempfile.TemporaryDirectory()))
            sibling=scratch/'keep';sibling.write_text('keep');callback=BoundedQuantizedReference(graph,scratch,output_chunk_rows=3)
            stack.callback(callback.close);state=graph.new_state(context);control=graph.new_state(context)
            stack.callback(state.close);stack.callback(control.close)
            def baseline(operation,value):
                projection=FullLinear(operation['id'],graph.projections[operation['id']].source,whole_executor,whole,12)
                projection.mode='clear-reference';identities[operation['id']]=projection.definition['weightId']
                return projection(value)
            quantize=callback.executor._per_channel_weight_quantization;shapes=[]
            def bounded(weight,*args):
                shapes.append(weight.shape);self.assertLessEqual(weight.shape[1],3)
                return quantize(weight,*args)
            with patch.object(callback.executor,'_per_channel_weight_quantization',side_effect=bounded):
                for values in ([1,2,3,4,5],[6],[7],[8]):
                    tokens=torch.tensor([values],dtype=torch.int64)
                    expected=graph(tokens,control,context,execution='verified-projections',verified_client_projection=baseline)
                    actual=graph(tokens,state,context,execution='verified-projections',verified_client_projection=callback)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    self.assertEqual(list(scratch.iterdir()),[sibling])
            ordered,identity=callback.inventory(graph.order)
            expected=[dict(name=name,weightId=identities[name]) for name in graph.order]
            self.assertEqual(ordered,expected);self.assertEqual(identity,digest(canonical(expected)))
            self.assertEqual(callback.materializations,len(graph.order)*4);self.assertTrue(shapes)

    def test_cancelled_or_insufficient_scratch_is_terminal_and_preserves_siblings(self):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_quantized_reference import BoundedQuantizedReference
        torch,graph,_,_=FlashModelReferenceTests().fixture();operation=graph.projections[graph.order[0]].operation
        for failure in ('cancel','space'):
            with self.subTest(failure=failure),ExitStack() as stack:
                scratch=Path(stack.enter_context(tempfile.TemporaryDirectory()));sibling=scratch/'keep';sibling.write_text('keep')
                event=threading.Event();callback=BoundedQuantizedReference(graph,scratch,cancelled=event,output_chunk_rows=1)
                quantize=callback.executor._per_channel_weight_quantization
                def cancel(*args):
                    result=quantize(*args);event.set();return result
                if failure=='cancel':stack.enter_context(patch.object(callback.executor,'_per_channel_weight_quantization',side_effect=cancel))
                else:stack.enter_context(patch('flash_quantized_reference.shutil.disk_usage',return_value=type('Usage',(),{'free':0})()))
                with self.assertRaises(InterruptedError if failure=='cancel' else ValueError):
                    callback(operation,torch.ones((1,1,operation['inputWidth'])))
                self.assertEqual(list(scratch.iterdir()),[sibling]);event.clear()
                with self.assertRaises(ValueError):callback(operation,torch.ones((1,1,operation['inputWidth'])))
                with self.assertRaises(ValueError):callback.inventory(graph.order)

    def test_quantizer_metadata_change_rejects_before_reference_product(self):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_quantized_reference import BoundedQuantizedReference
        torch,graph,_,_=FlashModelReferenceTests().fixture();operation=graph.projections[graph.order[0]].operation
        with tempfile.TemporaryDirectory() as scratch:
            callback=BoundedQuantizedReference(graph,scratch,output_chunk_rows=1);value=torch.ones((1,1,operation['inputWidth']))
            callback(operation,value);quantize=callback.executor._per_channel_weight_quantization
            def change(*args):
                coefficients,scale,zero,total=quantize(*args)
                return coefficients,scale+1,zero,total
            with patch.object(callback.executor,'_per_channel_weight_quantization',side_effect=change),patch('full_model.integer_dot') as product:
                with self.assertRaisesRegex(ValueError,'metadata changed'):callback(operation,value)
                product.assert_not_called()
            self.assertFalse(list(Path(scratch).iterdir()))


if __name__=='__main__':unittest.main()
