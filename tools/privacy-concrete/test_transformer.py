"""Opt-in actual checkpoint autoregression and client key-custody failure checks."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_MODEL_DIR'), 'opt-in pinned local model fixture')
class TransformerTests(unittest.TestCase):
    def test_multiple_encrypted_steps_with_fresh_keys_and_fixed_disclosure_for_two_seeds(self):
        from experiment import run
        from protocol import POLICY, digest
        reports = []
        for seed in (15496, 464):
            report = run(os.environ['DOPPLER_CONCRETE_MODEL_DIR'], seed)
            self.assertTrue(report['matchesQuantizedReference'])
            self.assertTrue(report['matchesFloatGreedyReference'])
            self.assertFalse(report['qualified'])
            self.assertFalse(report['productionBackendAvailable'])
            self.assertEqual(report['encryptedProjectionCalls'], POLICY['steps'] * POLICY['projectionCount'])
            self.assertEqual(len(report['clientLocalOutputTokenIds']), POLICY['steps'])
            for name, identity in report['sourceDigests'].items():
                self.assertEqual(identity, digest(Path(__file__).with_name(name).read_bytes()))
            reports.append(report)
            destination = os.environ.get('DOPPLER_CONCRETE_REPORT_DIR')
            if destination:
                directory = Path(destination)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'tiny-gpt2-seed-{seed}.json').write_text(json.dumps(report, indent=2) + '\n')
        self.assertEqual(reports[0]['modelId'], reports[1]['modelId'])
        self.assertNotEqual(reports[0]['keyContextId'], reports[1]['keyContextId'])
        self.assertEqual(reports[0]['sentBytes'], reports[1]['sentBytes'])
        self.assertEqual(reports[0]['receivedBytes'], reports[1]['receivedBytes'])

    def test_provider_startup_failure_releases_client_key_references(self):
        import experiment
        actual_factory = experiment.GLWELinearLayerExecutor
        executors = []
        def create(**options):
            executor = actual_factory(**options)
            executors.append(executor)
            return executor
        with patch.object(experiment, 'GLWELinearLayerExecutor', side_effect=create), patch.object(experiment, 'EvaluatorBridge', side_effect=RuntimeError('startup failed')):
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                experiment.run(os.environ['DOPPLER_CONCRETE_MODEL_DIR'])
        self.assertEqual(len(executors), 1)
        self.assertIsNone(executors[0].private_key)
        self.assertIsNone(executors[0].compression_key)


if __name__ == '__main__':
    unittest.main()
