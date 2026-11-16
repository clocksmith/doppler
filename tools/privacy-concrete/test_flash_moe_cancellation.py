"""Actual MoE control flow with simulated tensors/crypto; no native qualification."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock,patch


class Value:
    def reshape(self,*args):return self
    def tolist(self):return [1.,2.]


class FlashMoECancellationTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('_tested_cancel_moe',Path(__file__).with_name('flash_moe.py'))
        self.module=importlib.util.module_from_spec(spec)
        native=SimpleNamespace();torch=SimpleNamespace(nn=SimpleNamespace(),no_grad=nullcontext,tensor=lambda *args,**kwargs:Value(),float32='float32')
        with patch.dict(sys.modules,{'torch':torch,'concrete_ml_extensions':native,
            'concrete.ml.quantization.linear_op_glwe_backend':SimpleNamespace(GLWELinearLayerExecutor=object),
            'full_model':SimpleNamespace(FullLinear=object),'full_bridge':SimpleNamespace(FullBridge=object)}):spec.loader.exec_module(self.module)
        self.event=threading.Event();self.model=self.module.StreamedMoE.__new__(self.module.StreamedMoE)
        self.rows=[dict(name='shared/gate',inputWidth=2,outputWidth=2),dict(name='shared/up',inputWidth=2,outputWidth=2)]
        self.plan=SimpleNamespace(definition=lambda:dict(projections=self.rows))
        self.model.executor=SimpleNamespace(glwe_crypto_params=object());self.model.prepared={}

    def test_cancelled_constructor_and_run_cannot_prepare_backend_or_open_source(self):
        self.event.set()
        with patch.object(self.module,'moe_backend') as backend,patch.object(self.module,'FlashSource') as source:
            with self.assertRaises(InterruptedError):self.module.StreamedMoE('unused','unused',cancelled=self.event)
            with self.assertRaises(InterruptedError):self.module.run('unused','unused','encrypted',cancelled=self.event)
            backend.assert_not_called();source.assert_not_called()

    def test_cancelled_preparation_publishes_no_partial_metadata_and_leaves_cache_files(self):
        with __import__('tempfile').TemporaryDirectory() as directory:
            cache=Path(directory)/'existing-public-cache';cache.write_bytes(b'public coefficient canary')
            def projection(name):
                self.event.set()
                return SimpleNamespace(definition=self.rows[0],release_float_reference=Mock())
            self.model.projection=Mock(side_effect=projection)
            with self.assertRaises(InterruptedError):self.model.prepare_projections(self.rows,cancelled=self.event)
            self.assertEqual(self.model.projection.call_count,1);self.assertEqual(self.model.prepared,{})
            self.assertEqual(cache.read_bytes(),b'public coefficient canary')

    def test_cancelled_evaluation_cannot_read_plan_or_materialize_a_projection(self):
        self.event.set();self.model.projection=Mock()
        plan=SimpleNamespace(definition=Mock(side_effect=AssertionError('cancelled plan read')))
        with self.assertRaises(InterruptedError):self.model.evaluate([1.,2.],'clear-reference',plan,cancelled=self.event)
        self.model.projection.assert_not_called();plan.definition.assert_not_called()

    def test_cancellation_inside_projection_clears_attempt_bindings_and_stops_later_hops(self):
        owner=self
        class Projection:
            definition=owner.rows[0]
            def __call__(self,value):owner.event.set();return Value()
        projection=Projection();self.model.projection=Mock(return_value=projection)
        def run(hidden,project):
            for row in self.rows:project(dict(id=row['name']),hidden)
            return [1.,2.]
        self.module.PROGRAM=SimpleNamespace(attempt=lambda:SimpleNamespace(run=run))
        with self.assertRaises(InterruptedError):self.model.evaluate([1.,2.],'clear-reference',self.plan,cancelled=self.event)
        self.assertEqual(self.model.projection.call_count,1)
        self.assertIsNone(projection.bridge);self.assertIsNone(projection.secret);self.assertIsNone(projection.parameters)

    def test_encrypted_signal_mismatch_or_missing_execution_context_rejects_before_plan(self):
        self.model.projection=Mock();plan=SimpleNamespace(definition=Mock(side_effect=AssertionError('invalid context read plan')))
        bridge=SimpleNamespace(cancelled=threading.Event(),check_active=Mock())
        for mode,connection,key in [('execute',None,None),('execute',bridge,object()),('clear-reference',bridge,object()),('unsupported',None,None)]:
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                self.model.evaluate([1.,2.],mode,plan,connection,key,cancelled=self.event)
        self.model.projection.assert_not_called();plan.definition.assert_not_called();bridge.check_active.assert_not_called()

    def test_bridge_observes_cancellation_terminally_before_plan_or_projection(self):
        from test_full_bridge import FullBridgeOrderingTests
        helper=FullBridgeOrderingTests();helper.setUp()
        bridge=helper.bridge;bridge.cancelled=self.event;self.event.set();self.model.projection=Mock()
        with self.assertRaises(InterruptedError):self.model.evaluate([1.,2.],'execute',self.plan,bridge,object(),cancelled=self.event)
        self.event.clear()
        with self.assertRaises(ValueError):self.model.evaluate([1.,2.],'execute',self.plan,bridge,object(),cancelled=self.event)
        self.model.projection.assert_not_called()

    def test_public_source_constructor_propagates_cancellation_to_existing_verified_loader(self):
        from flash_weights import FlashSource
        self.event.set()
        with patch('flash_weights.verify_blob') as verify:
            with self.assertRaises(InterruptedError):FlashSource('/nonexistent-unused',cancelled=self.event)
            verify.assert_not_called()


if __name__=='__main__':unittest.main()
