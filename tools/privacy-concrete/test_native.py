"""Opt-in real GLWE transport tests. No model download, simulation or GPU required."""
import os
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest


@unittest.skipUnless(os.environ.get('DOPPLER_CONCRETE_NATIVE') == '1', 'opt-in pinned Concrete native dependencies')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import concrete_ml_extensions as ext
        import numpy as np
        from protocol import POLICY, digest
        cls.ext, cls.np = ext, np
        cls.parameters = ext.MatmulCryptoParameters.deserialize(ext.default_params())
        cls.secret, cls.key = ext.cpu_create_private_key(cls.parameters)
        cls.matrices = {}
        width = POLICY['matrixWidth']
        for index in range(POLICY['projectionCount']):
            matrix = np.full((width, width), index + 1, dtype='<u8')
            cls.matrices[digest(matrix.tobytes())] = matrix

    def input(self, seed):
        from protocol import POLICY
        data = self.np.full((POLICY['contextLength'], POLICY['matrixWidth']), seed, dtype=self.np.uint64)
        return self.ext.encrypt_matrix(pkey=self.secret, crypto_params=self.parameters, data=data)

    def bridge(self, **options):
        from bridge import EvaluatorBridge
        return EvaluatorBridge('a' * 64, self.key, self.matrices, **options)

    def test_real_serialized_evaluation_and_public_replay(self):
        evaluator = self.bridge()
        try:
            matrix = next(iter(self.matrices.values()))
            encrypted = self.input(128)
            result = evaluator.matrix_multiplication(encrypted, matrix, self.key)
            # Decryption occurs after byte equality with the public replay, never before it.
            decoded = self.ext.decrypt_matrix(result, self.secret, self.parameters, matrix.shape[1])
            from protocol import POLICY
            self.assertEqual(decoded.shape, (POLICY['contextLength'], POLICY['matrixWidth']))
            self.assertLessEqual(int(self.np.abs(decoded.astype(self.np.int64) - 128 * POLICY['matrixWidth']).max()), 64)
            self.assertEqual(evaluator.calls, 1)
            self.assertGreater(evaluator.replay_seconds, 0)
        finally:
            evaluator.close()
        self.assertIsNotNone(evaluator.child.returncode)

    def test_valid_malicious_evaluation_rejects_even_for_zero_plaintext(self):
        counts = []
        for seed in (0, 1):
            # Same adversarial computation uses only the wire ciphertext, public weights and evaluation key.
            encrypted = self.input(seed)
            captured = self.ext.EncryptedMatrix.deserialize(encrypted.serialize())
            matrix = next(iter(self.matrices.values()))
            def substitute(_value):
                return self.ext.cpu_matrix_multiplication(captured, matrix + 1, self.key).serialize()
            evaluator = self.bridge(mutate=substitute)
            try:
                with self.assertRaisesRegex(ValueError, 'before decryption'):
                    evaluator.matrix_multiplication(encrypted, matrix, self.key)
                counts.append(evaluator.calls)
            finally:
                evaluator.close()
        self.assertEqual(counts, [1, 1])

    def test_cancellation_stops_before_request_and_closes_provider(self):
        cancelled = threading.Event()
        evaluator = self.bridge(cancelled=cancelled)
        cancelled.set()
        try:
            with self.assertRaises(InterruptedError):
                evaluator.matrix_multiplication(self.input(1), next(iter(self.matrices.values())), self.key)
            self.assertEqual(evaluator.calls, 0)
        finally:
            evaluator.close()
        self.assertIsNotNone(evaluator.child.returncode)

    def test_unsupported_model_and_seed_fail_before_any_provider_or_key_generation(self):
        from experiment import run
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'model.safetensors').write_bytes(b'unsupported fixture')
            (directory / 'config.json').write_text('{}')
            with patch('bridge.subprocess.Popen') as provider, patch('concrete_ml_extensions.cpu_create_private_key') as keygen:
                with self.assertRaisesRegex(ValueError, 'Unsupported fixture seed'):
                    run(directory, seed=-1)
                with self.assertRaisesRegex(ValueError, 'Pinned model fixture'):
                    run(directory)
                provider.assert_not_called()
                keygen.assert_not_called()

    def test_disappearing_evaluator_fails_without_retry_or_plaintext_fallback(self):
        evaluator = self.bridge()
        evaluator.child.kill()
        evaluator.child.wait()
        try:
            with self.assertRaises((BrokenPipeError, ValueError)):
                evaluator.matrix_multiplication(self.input(1), next(iter(self.matrices.values())), self.key)
            self.assertEqual(evaluator.calls, 0)
        finally:
            evaluator.close()


if __name__ == '__main__':
    unittest.main()
