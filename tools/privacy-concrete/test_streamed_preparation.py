"""Real source/cache bytes with simulated quantization, not native cryptography."""
import gc,os,tempfile,threading,unittest,weakref
from unittest.mock import patch
import test_prepare_cancellation as preparation
import test_flash_streamed_moe as streamed

@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in Torch reference')
class StreamedPreparationTests(unittest.TestCase):
    def fixture(self,directory):
        import flash_streamed_moe
        helper=preparation.PreparationCancellationTests();helper.setUp()
        torch,moe,_,source,_=streamed.FlashStreamedMoETests().fixture(directory)
        class Quantizer:
            def _per_channel_weight_quantization(self,weight,*args):
                q=torch.from_numpy(weight.copy()).mul(100).round()
                return q,torch.ones(weight.shape[1]),torch.zeros(weight.shape[1]),q.sum(0)
        return helper.module,torch,moe.ops['expert_1_up'],source,Quantizer()

    def test_meta_source_matches_materialized_coefficients_and_releases_weights(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as cache:
            model,torch,op,source,executor=self.fixture(directory)
            with op.materialized_weight() as weight:
                reference=torch.nn.Linear(op.in_features,op.out_features,bias=False)
                reference.weight=torch.nn.Parameter(weight.clone(),requires_grad=False)
            del weight
            leases=[];read=source.tensor
            def tracked(*args,**kwargs):
                value=read(*args,**kwargs);leases.append(weakref.ref(value));return value
            with patch.object(source,'tensor',side_effect=tracked):actual=model.FullLinear('streamed',op,executor,cache)
            expected=model.FullLinear('materialized',reference,executor,cache)
            self.assertEqual(actual.definition['weightId'],expected.definition['weightId'])
            self.assertTrue(op.weight.is_meta);gc.collect();self.assertTrue(all(ref() is None for ref in leases))
            actual.release_float_reference();borrowed=actual.prepared_copy()
            with borrowed.coefficient_matrix() as left,expected.coefficient_matrix() as right:self.assertTrue((left==right).all())

    def test_cancel_or_changed_source_cannot_publish_coefficients(self):
        for changed in ['cancel','close']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as cache:
                model,torch,op,source,executor=self.fixture(directory);event=threading.Event()
                quantize=executor._per_channel_weight_quantization
                def mutate(*args):
                    result=quantize(*args)
                    if changed=='cancel':event.set()
                    else:source.closed=True
                    return result
                executor._per_channel_weight_quantization=mutate
                with patch.object(model,'store_public_matrix') as store:
                    with self.assertRaises((ValueError,InterruptedError)):model.FullLinear('streamed',op,executor,cache,cancelled=event)
                    store.assert_not_called()

    def test_precancelled_never_reads_source(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as cache:
            model,torch,op,source,executor=self.fixture(directory);event=threading.Event();event.set()
            with patch.object(source,'tensor') as read:
                with self.assertRaises(InterruptedError):model.FullLinear('streamed',op,executor,cache,cancelled=event)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
