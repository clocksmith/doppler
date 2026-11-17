import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from client_progress import ClientProgress


class ClientProgressTests(unittest.TestCase):
    def test_sparse_records_never_copy_observer_fields_or_claim_early_acceptance(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'status.jsonl'
            with ClientProgress(path) as progress:
                self.assertEqual(path.stat().st_mode&0o777,0o600)
                progress.bind('a'*64,513)
                for sequence in range(1,514):
                    progress.observe('received-tile',dict(sequence=sequence,prompt='PRIVATE-PROMPT',
                        projection='PRIVATE-OUTPUT',ciphertext='PRIVATE-CIPHER',integrity='PRIVATE-CHECK'))
                records=[json.loads(row) for row in path.read_text().splitlines()]
                self.assertEqual([r['phase'] for r in records],['planned','receiving','receiving','receiving'])
                self.assertEqual([r['receivedTiles'] for r in records],[0,256,512,513])
                self.assertNotIn('PRIVATE',path.read_text())
                progress.finish(prepare_only=False)
                with self.assertRaises(ValueError):progress.observe('received-tile',dict(sequence=514))
            self.assertEqual(json.loads(path.read_text().splitlines()[-1])['phase'],'validated')

    def test_incomplete_reordered_unknown_or_unbound_events_never_record_validation(self):
        with tempfile.TemporaryDirectory() as root:
            for kind,value in [('received-tile',{'sequence':True}),('received-tile',{'sequence':2}),('private-result',{'sequence':1})]:
                path=Path(root)/('status-'+str(len(list(Path(root).iterdir()))))
                with ClientProgress(path) as progress:
                    with self.assertRaises(ValueError):progress.observe(kind,value)
                    progress.bind('b'*64,3)
                    with self.assertRaises(ValueError):progress.observe(kind,value)
                    with self.assertRaises(ValueError):progress.finish(prepare_only=False)
                self.assertNotIn('validated',path.read_text())

    def test_preparation_has_zero_received_tiles_and_its_own_completion_label(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'status'
            with ClientProgress(path) as progress:
                progress.bind('c'*64,603472);progress.finish(prepare_only=True)
            record=json.loads(path.read_text().splitlines()[-1]);self.assertEqual(record['phase'],'prepared')
            self.assertEqual(record['receivedTiles'],0);self.assertNotIn('validated',path.read_text())

    def test_readiness_has_no_inference_tiles_and_never_records_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'status'
            with ClientProgress(path) as progress:
                progress.bind('c'*64,603472)
                with self.assertRaises(ValueError):progress.finish(prepare_only=True,readiness_only=True)
                with self.assertRaises(ValueError):progress.finish(prepare_only=False,readiness_only=1)
                progress.finish(prepare_only=False,readiness_only=True)
            record=json.loads(path.read_text().splitlines()[-1]);self.assertEqual(record['phase'],'provider-ready')
            self.assertEqual(record['receivedTiles'],0);self.assertNotIn('validated',path.read_text())
            path=Path(root)/'consumed'
            with ClientProgress(path) as progress:
                progress.bind('d'*64,1);progress.observe('received-tile',dict(sequence=1))
                with self.assertRaises(ValueError):progress.finish(prepare_only=False,readiness_only=True)

    def test_cli_readiness_records_only_provider_ready_after_return_and_cleanup(self):
        import flash_generate_run as runner
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);source=root/'source';source.mkdir();scratch=root/'scratch';scratch.mkdir();status=scratch/'status'
            args=['--source',str(source),'--reference',str(root/'unused.py'),'--scratch',str(scratch),
                '--backend-profile','flash-text-all-linear-ntt12-rebuild-prefill5-check','--tenant-id','fixture',
                '--privacy-id','local','--lock',str(root/'lock'),'--pause-file',str(root/'pause'),
                '--resource-wait-seconds','1','--public-fixture','flash-public-capital-text','--status-file',str(status),'--readiness-only']
            def attempt(*args,**kwargs):
                self.assertTrue(kwargs['readiness_only']);self.assertFalse(kwargs['prepare_only'])
                kwargs['progress'].bind('c'*64,603472)
                self.assertNotIn('provider-ready',status.read_text())
                return dict(productionBackendAvailable=False,securityQualified=False)
            with patch.object(runner,'run',side_effect=attempt),patch('sys.stdout',io.StringIO()),patch('sys.stderr',io.StringIO()):
                self.assertEqual(runner.main(args),0)
            self.assertEqual(json.loads(status.read_text().splitlines()[-1])['phase'],'provider-ready')
            self.assertNotIn('validated',status.read_text())

    def test_existing_files_and_symlinks_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            existing=Path(root)/'keep';existing.write_text('keep');link=Path(root)/'link';link.symlink_to(existing)
            for path in (existing,link):
                with self.assertRaises(FileExistsError):ClientProgress(path)
            self.assertEqual(existing.read_text(),'keep');self.assertTrue(link.is_symlink())

    def test_invalid_binding_and_closed_writer_reject_without_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            for plan_id,total in [('unsafe',3),('a'*64,True),('a'*64,0),('a'*64,2**31)]:
                path=Path(root)/str(len(list(Path(root).iterdir())))
                with ClientProgress(path) as progress:
                    with self.assertRaises(ValueError):progress.bind(plan_id,total)
                self.assertEqual(path.read_bytes(),b'')
                with self.assertRaises(ValueError):progress.bind('a'*64,3)

    def test_cli_records_validation_only_after_successful_run_and_cleanup(self):
        import flash_generate_run as runner
        from full_status import ExecutionRejected
        for rejected in (False,True):
            with self.subTest(rejected=rejected),tempfile.TemporaryDirectory() as root:
                root=Path(root);source=root/'source';source.mkdir();scratch=root/'scratch';scratch.mkdir();status=scratch/'status'
                args=['--source',str(source),'--reference',str(root/'unused.py'),'--scratch',str(scratch),
                    '--backend-profile','flash-text-all-linear-ntt12-rebuild-prefill5-check','--tenant-id','fixture',
                    '--privacy-id','local','--lock',str(root/'lock'),'--pause-file',str(root/'pause'),
                    '--resource-wait-seconds','1','--public-fixture','flash-public-capital-text','--status-file',str(status)]
                def attempt(*args,**kwargs):
                    progress=kwargs['progress'];progress.bind('f'*64,1)
                    progress.observe('received-tile',dict(sequence=1,prompt='PRIVATE'))
                    self.assertNotIn('validated',status.read_text())
                    if rejected:raise ExecutionRejected('integrity-check-failed')
                    return dict(productionBackendAvailable=False,securityQualified=False)
                with patch.object(runner,'run',side_effect=attempt),patch('sys.stdout',io.StringIO()),patch('sys.stderr',io.StringIO()):
                    self.assertEqual(runner.main(args),1 if rejected else 0)
                self.assertEqual('validated' in status.read_text(),not rejected)
                self.assertNotIn('PRIVATE',status.read_text())

    def test_cli_rejects_progress_outside_scratch_before_creating_a_file_or_running(self):
        import flash_generate_run as runner
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);source=root/'source';source.mkdir();scratch=root/'scratch';scratch.mkdir();status=source/'unsafe-status'
            args=['--source',str(source),'--reference',str(root/'unused.py'),'--scratch',str(scratch),
                '--backend-profile','flash-text-all-linear-ntt12-rebuild-prefill5-check','--tenant-id','fixture',
                '--privacy-id','local','--lock',str(root/'lock'),'--pause-file',str(root/'pause'),
                '--resource-wait-seconds','1','--public-fixture','flash-public-capital-text','--status-file',str(status)]
            with patch.object(runner,'run') as execute,patch('sys.stdout',io.StringIO()),patch('sys.stderr',io.StringIO()):
                self.assertEqual(runner.main(args),1);execute.assert_not_called()
            self.assertFalse(status.exists())


if __name__=='__main__':unittest.main()
