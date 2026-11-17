"""Bounded floating quality oracle; tiny model/source, no native encryption."""
import os
import threading
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch/source reference')
class BoundedFloatTests(unittest.TestCase):
    def test_complete_five_row_prefill_and_three_decode_steps_match_whole_reference(self):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_float_reference import BoundedFloatingReference
        torch,graph,_,context=FlashModelReferenceTests().fixture()
        whole=graph.new_state(context);bounded=graph.new_state(context);callback=BoundedFloatingReference(graph,output_chunk_rows=2)
        try:
            for values in ([1,2,3,4,5],[6],[7],[8]):
                tokens=torch.tensor([values],dtype=torch.int64)
                expected=graph(tokens,whole,context,execution='public-reference')
                actual=graph(tokens,bounded,context,execution='verified-projections',verified_client_projection=callback)
                torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-5)
                self.assertEqual(int(actual[0,-1].argmax()),int(expected[0,-1].argmax()))
            self.assertLessEqual(callback.peak_fp32_weight_bytes,max(graph.projections[name].operation['inputWidth']*8 for name in graph.order))
            self.assertEqual(callback.operations,len(graph.order)*4)
        finally:whole.close();bounded.close();callback.close()

    def test_cancelled_chunk_is_terminal_before_later_source_rows(self):
        from test_flash_model_reference import FlashModelReferenceTests
        from flash_float_reference import BoundedFloatingReference
        import torch.nn.functional as functional
        torch,graph,_,_=FlashModelReferenceTests().fixture();event=threading.Event()
        callback=BoundedFloatingReference(graph,cancelled=event,output_chunk_rows=1)
        operation=graph.projections[graph.order[0]].operation;calls=[];linear=functional.linear
        def cancel(*args):
            calls.append(True);result=linear(*args);event.set();return result
        with patch.object(functional,'linear',side_effect=cancel):
            with self.assertRaises(InterruptedError):callback(operation,torch.ones((1,1,operation['inputWidth'])))
        self.assertEqual(len(calls),1);event.clear()
        with self.assertRaises(ValueError):callback(operation,torch.ones((1,1,operation['inputWidth'])))


if __name__=='__main__':unittest.main()
