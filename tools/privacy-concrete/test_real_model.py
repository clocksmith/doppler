"""Opt-in exact small-Qwen generation; no download or production privacy claim."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('DOPPLER_QWEN_MODEL_DIR'), 'opt-in pinned local Qwen source')
class QwenTests(unittest.TestCase):
    def test_two_prompts_use_fresh_keys_fixed_wire_and_exact_selected_projection(self):
        from real_model import run
        from protocol import digest
        reports = []
        for label, prompt in [('capital', 'The capital of France is'), ('computer', 'A computer is')]:
            report = run(os.environ['DOPPLER_QWEN_MODEL_DIR'], prompt)
            self.assertFalse(report['qualified'])
            self.assertFalse(report['productionBackendAvailable'])
            self.assertFalse(report['fullModelEncrypted'])
            self.assertTrue(report['matchesQuantizedGreedyReference'])
            self.assertTrue(report['matchesFloatGreedyReference'])
            self.assertEqual(report['encryptedProjectionCalls'], 4)
            self.assertEqual(report['projection']['name'], 'model.layers.0.self_attn.k_proj')
            for name, identity in report['sourceDigests'].items():
                self.assertEqual(identity, digest(Path(__file__).with_name(name).read_bytes()))
            reports.append(report)
            destination = os.environ.get('DOPPLER_QWEN_REPORT_DIR')
            if destination:
                directory = Path(destination)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'qwen05b-{label}.json').write_text(json.dumps(report, indent=2) + '\n')
        self.assertEqual(reports[0]['modelId'], reports[1]['modelId'])
        self.assertNotEqual(reports[0]['keyContextId'], reports[1]['keyContextId'])
        self.assertEqual(reports[0]['sentBytes'], reports[1]['sentBytes'])
        self.assertEqual(reports[0]['receivedBytes'], reports[1]['receivedBytes'])

    def test_unsupported_source_and_profile_fail_before_key_generation_or_provider(self):
        from real_model import run
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'model.safetensors').write_bytes(b'unsupported source')
            with patch('real_model.EvaluatorBridge') as provider, patch('concrete_ml_extensions.cpu_create_private_key') as keygen:
                with self.assertRaisesRegex(ValueError, 'Pinned real-model artifact'):
                    run(temporary, 'public fixture')
                with self.assertRaisesRegex(ValueError, 'Unsupported installed'):
                    run(temporary, 'public fixture', profile_name='Qwen3.8-Flash-Next')
                provider.assert_not_called()
                keygen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
