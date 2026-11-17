"""Resource/ownership tests; mocked model/native lifecycle is not physical evidence."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flash_generate_run import resource_turn,run
from plan_backends import REBUILD_PREFILL5_PROFILE
from protocol import require_active


class ResourceTurnTests(unittest.TestCase):
    def test_existing_pause_prevents_any_checkpoint_or_native_access_and_is_preserved(self):
        import flash_text_rebuild5_plan as backend
        with tempfile.TemporaryDirectory() as root:
            lock=Path(root)/'lock';lock.touch();pause=Path(root)/'pause';pause.write_text('other-owner')
            with patch.dict(os.environ,{'OMP_NUM_THREADS':'4','RAYON_NUM_THREADS':'4'}),patch.object(backend,'versions') as native,patch.object(Path,'is_dir') as source:
                with self.assertRaises(TimeoutError):run('/unread/source','/unread/reference','/unread/scratch','public',
                    backend_profile=REBUILD_PREFILL5_PROFILE,tenant_id='tenant',privacy_id='restricted',
                    lock_path=lock,pause_path=pause,resource_wait_seconds=.01)
                source.assert_not_called();native.assert_not_called()
            self.assertEqual(pause.read_text(),'other-owner')

    def test_new_pause_cancels_attempt_and_unlocks_without_removing_pause(self):
        with tempfile.TemporaryDirectory() as root:
            lock=Path(root)/'lock';lock.touch();pause=Path(root)/'pause'
            with resource_turn(lock,pause,1) as signal:
                self.assertFalse(signal.is_set());pause.write_text('other-owner')
                with self.assertRaises(InterruptedError):require_active(signal)
            self.assertEqual(pause.read_text(),'other-owner')
            pause.unlink() # This test owns its fixture, not the real cooperative pause.
            with resource_turn(lock,pause,1) as signal:self.assertFalse(signal.is_set())

    def test_resource_contract_does_not_create_an_unagreed_lock(self):
        with tempfile.TemporaryDirectory() as root:
            lock=Path(root)/'absent';pause=Path(root)/'pause'
            with self.assertRaises(ValueError):
                with resource_turn(lock,pause,1):self.fail('unagreed lock admitted')
            self.assertFalse(lock.exists())


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in installed Torch/source interfaces')
class RunnerLifecycleTests(unittest.TestCase):
    def test_provider_stops_before_public_leases_and_owned_scratch_are_released(self):
        from contextlib import ExitStack
        import concrete_ml_extensions as ext
        import flash_text_rebuild5_plan as backend
        import torch
        events=[];original_threads=torch.get_num_threads();temporary_type=tempfile.TemporaryDirectory
        class Temporary(temporary_type):
            def cleanup(self):events.append('scratch-close');super().cleanup()
        def stage(*args,**kwargs):
            return SimpleNamespace(spec={'wholeShardBytes':1},close=lambda:events.append('source-close'))
        prepared=SimpleNamespace(order=['projection'],preparation_seconds=0.,rematerialization_seconds=0.)
        graph=SimpleNamespace(identity='f'*64)
        bridge=SimpleNamespace(key_id='key',session='request',close=lambda:events.append('provider-close'))
        with temporary_type() as root,ExitStack() as stack:
            root=Path(root);source=root/'source';source.mkdir();scratch=root/'scratch';scratch.mkdir()
            sibling=scratch/'keep';sibling.write_text('keep');lock=root/'lock';lock.touch();pause=root/'pause'
            reference=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
            stack.enter_context(patch.dict(os.environ,{'OMP_NUM_THREADS':'4','RAYON_NUM_THREADS':'4'}))
            stack.enter_context(patch.object(backend,'versions',return_value={'fixture':'mocked lifecycle'}))
            stack.enter_context(patch.object(backend,'write_plan',return_value=(object(),object())))
            stack.enter_context(patch('flash_generate_run.shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)))
            stack.enter_context(patch('flash_generate_run.tempfile.TemporaryDirectory',Temporary))
            stack.enter_context(patch('flash_tokenizer.encode',return_value=(1,2,3,4,5)))
            stack.enter_context(patch('flash_stage_source.FlashStageSource',side_effect=stage))
            stack.enter_context(patch('local_token_rows.LocalTokenRows.flash',side_effect=stage))
            stack.enter_context(patch('local_ngram_rows.LocalNgramRows.flash',side_effect=stage))
            stack.enter_context(patch('flash_graph_source.bind_graph',return_value=graph))
            stack.enter_context(patch('flash_prepared_graph.PreparedFlashGraph',return_value=prepared))
            stack.enter_context(patch('research_parameters.component_parameters',return_value={'parameters':{}}))
            stack.enter_context(patch.object(ext,'MatmulCryptoParameters',SimpleNamespace(deserialize=lambda value:object())))
            stack.enter_context(patch.object(ext,'cpu_create_private_key',return_value=(object(),object()),create=True))
            stack.enter_context(patch('concrete.ml.quantization.linear_op_glwe_backend.GLWELinearLayerExecutor',return_value=SimpleNamespace()))
            stack.enter_context(patch('checked_bridge.CheckedBridge',return_value=bridge))
            stack.enter_context(patch('flash_generation.generate',side_effect=RuntimeError('mocked generation failure')))
            with self.assertRaisesRegex(RuntimeError,'mocked generation failure'):
                run(source,reference,scratch,'public fixture',backend_profile=REBUILD_PREFILL5_PROFILE,tenant_id='tenant',privacy_id='restricted',
                    lock_path=lock,pause_path=pause,resource_wait_seconds=1)
            self.assertLess(events.index('provider-close'),events.index('source-close'))
            self.assertLess(events.index('source-close'),events.index('scratch-close'))
            self.assertEqual(list(scratch.iterdir()),[sibling]);self.assertEqual(sibling.read_text(),'keep')
        self.assertEqual(torch.get_num_threads(),original_threads)


if __name__=='__main__':unittest.main()
