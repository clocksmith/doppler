"""Resource/ownership tests; mocked model/native lifecycle is not physical evidence."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flash_generate_run import resource_turn,run,require_private_resource_scope
from plan_backends import REBUILD_PREFILL5_PROFILE
from protocol import require_active


class ResourceTurnTests(unittest.TestCase):
    def test_kernel_scope_rejects_shared_unbounded_swappable_or_non_group_kill_jobs(self):
        import flash_generate_run
        for change in (None,'shared','memory','swap','oom','cpu'):
            with self.subTest(change=change):
                values={'/proc/self/cgroup':'0::/a2-fixture.scope\n',
                    '/sys/fs/cgroup/a2-fixture.scope/cgroup.procs':str(os.getpid())+'\n',
                    '/sys/fs/cgroup/a2-fixture.scope/memory.max':str(8*1024**3),
                    '/sys/fs/cgroup/a2-fixture.scope/memory.swap.max':'0',
                    '/sys/fs/cgroup/a2-fixture.scope/memory.oom.group':'1',
                    '/sys/fs/cgroup/a2-fixture.scope/cpu.max':'800000 100000'}
                if change=='shared':values['/sys/fs/cgroup/a2-fixture.scope/cgroup.procs']+='999999\n'
                if change=='memory':values['/sys/fs/cgroup/a2-fixture.scope/memory.max']='max'
                if change=='swap':values['/sys/fs/cgroup/a2-fixture.scope/memory.swap.max']='1000'
                if change=='oom':values['/sys/fs/cgroup/a2-fixture.scope/memory.oom.group']='0'
                if change=='cpu':values['/sys/fs/cgroup/a2-fixture.scope/cpu.max']='900000 100000'
                with patch.object(flash_generate_run.Path,'read_text',lambda path:values[str(path)]):
                    if change is not None:
                        with self.assertRaises(ValueError):require_private_resource_scope()
                    else:self.assertEqual(require_private_resource_scope()['memoryMaxBytes'],8*1024**3)

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
        self._lifecycle(False)

    def test_reference_mismatch_releases_sources_without_creating_keys_or_provider(self):
        self._lifecycle(True)

    def test_preparation_only_returns_a_plan_without_creating_keys_or_provider(self):
        self._lifecycle(False,prepare_only=True)

    def test_rejected_reference_cannot_bind_a_client_progress_route(self):
        self._lifecycle(True,log=True)

    def _lifecycle(self,blocked_reference,prepare_only=False,log=False):
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
            progress=None
            if log:
                from client_progress import ClientProgress
                progress=stack.enter_context(ClientProgress(scratch/'status'))
            reference=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])/'modeling_qwen4_exp.py'
            stack.enter_context(patch.dict(os.environ,{'OMP_NUM_THREADS':'4','RAYON_NUM_THREADS':'4'}))
            stack.enter_context(patch.object(backend,'versions',return_value={'fixture':'mocked lifecycle'}))
            stack.enter_context(patch.object(backend,'write_plan',return_value=(SimpleNamespace(identity='plan',schedule=lambda:['tile']),object())))
            stack.enter_context(patch('flash_generate_run.shutil.disk_usage',return_value=SimpleNamespace(free=10*1024**3)))
            stack.enter_context(patch('flash_generate_run.require_private_resource_scope',return_value={'memoryMaxBytes':8*1024**3}))
            stack.enter_context(patch('flash_generate_run.tempfile.TemporaryDirectory',Temporary))
            stack.enter_context(patch('flash_tokenizer.encode',return_value=(1,2,3,4,5)))
            stack.enter_context(patch('flash_stage_source.FlashStageSource',side_effect=stage))
            stack.enter_context(patch('local_token_rows.LocalTokenRows.flash',side_effect=stage))
            stack.enter_context(patch('local_ngram_rows.LocalNgramRows.flash',side_effect=stage))
            stack.enter_context(patch('flash_graph_source.bind_graph',return_value=graph))
            stack.enter_context(patch('flash_prepared_graph.PreparedFlashGraph',return_value=prepared))
            stack.enter_context(patch('flash_checkpoint_reference.require_checkpoint_reference',return_value={'fixture':'mocked lifecycle'},
                side_effect=ValueError('reference mismatch') if blocked_reference else None))
            stack.enter_context(patch('flash_float_reference.BoundedFloatingReference',return_value=SimpleNamespace(close=lambda:None)))
            memory=stack.enter_context(patch('resource_memory.ResourceMeter',return_value=SimpleNamespace(
                close=lambda:events.append('meter-close'),mark=lambda phase:None,start=lambda:None)))
            stack.enter_context(patch('research_parameters.component_parameters',return_value={'parameters':{}}))
            stack.enter_context(patch.object(ext,'MatmulCryptoParameters',SimpleNamespace(deserialize=lambda value:object())))
            keygen=stack.enter_context(patch.object(ext,'cpu_create_private_key',return_value=(object(),object()),create=True))
            stack.enter_context(patch('concrete.ml.quantization.linear_op_glwe_backend.GLWELinearLayerExecutor',return_value=SimpleNamespace()))
            provider=stack.enter_context(patch('checked_bridge.CheckedBridge',return_value=bridge))
            stack.enter_context(patch('flash_generation.generate',side_effect=RuntimeError('mocked generation failure')))
            def invoke():return run(source,reference,scratch,'public fixture',backend_profile=REBUILD_PREFILL5_PROFILE,tenant_id='tenant',privacy_id='restricted',
                lock_path=lock,pause_path=pause,resource_wait_seconds=1,prepare_only=prepare_only,progress=progress)
            if prepare_only:
                result=invoke();self.assertFalse(result['encryptedExecution']);self.assertEqual(result['encryptedTileCount'],0)
                self.assertEqual(result['plannedEncryptedTileCount'],4)
                self.assertFalse(result['productionBackendAvailable']);self.assertFalse(result['securityQualified'])
            else:
                with self.assertRaisesRegex(ValueError if blocked_reference else RuntimeError,
                        'reference mismatch' if blocked_reference else 'mocked generation failure'):invoke()
            if blocked_reference or prepare_only:
                keygen.assert_not_called();provider.assert_not_called();memory.assert_not_called()
            else:
                self.assertLess(events.index('provider-close'),events.index('meter-close'))
                self.assertLess(events.index('meter-close'),events.index('source-close'))
            self.assertLess(events.index('source-close'),events.index('scratch-close'))
            self.assertEqual(set(scratch.iterdir()),{sibling}|({scratch/'status'} if log else set()));self.assertEqual(sibling.read_text(),'keep')
            if log:self.assertEqual((scratch/'status').read_bytes(),b'')
        self.assertEqual(torch.get_num_threads(),original_threads)


if __name__=='__main__':unittest.main()
