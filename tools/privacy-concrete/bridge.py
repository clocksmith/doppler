"""Process transport and full public encrypted replay, before any response decryption."""
import hmac
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import concrete_ml_extensions as ext
from protocol import POLICY, check, digest, receive, send
from profiles import load_profile


class EvaluatorBridge:
    def __init__(self, model_id, compression_key, matrices, mutate=None, cancelled=None, profile=None):
        self.profile = load_profile() if profile is None else profile
        if len(matrices) != self.profile.projection_count:
            raise ValueError('Incomplete public projection set')
        for weight_id, matrix in matrices.items():
            if (matrix.shape != (self.profile.input_width, self.profile.output_width)
                    or digest(matrix.tobytes(order='C')) != weight_id
                    or self.profile.expected_weight_ids and weight_id not in self.profile.expected_weight_ids):
                raise ValueError('Unsupported public projection identity or geometry')
        self.model_id, self.session = model_id, uuid.uuid4().hex
        self.key_bytes = compression_key.serialize()
        self.key_id = digest(self.key_bytes)
        self.key, self.matrices = compression_key, matrices
        self.mutate, self.cancelled = mutate, cancelled
        self.sequence = self.sent_bytes = self.received_bytes = self.calls = 0
        self.replay_seconds = self.remote_seconds = 0.0
        self.child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('provider.py')), model_id, self.profile.name],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      bufsize=0, env={'RAYON_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
        try:
            self.exchange('evaluation-key', self.key_bytes)
            for weight_id, matrix in matrices.items():
                self.exchange('public-weight', matrix.tobytes(order='C'), weight_id)
        except BaseException:
            self.close()
            raise

    def exchange(self, kind, payload, weight_id=''):
        if self.cancelled is not None and self.cancelled.is_set():
            raise InterruptedError('Encrypted experiment cancelled')
        header = self.profile.frame(self.model_id, self.session, self.key_id, self.sequence, kind, weight_id)
        self.sent_bytes += send(self.child.stdin, header, payload, cancelled=self.cancelled)
        response, result, size = receive(self.child.stdout, cancelled=self.cancelled)
        self.received_bytes += size
        check(response, self.profile.frame(self.model_id, self.session, self.key_id, self.sequence,
                                'result' if kind == 'evaluate' else 'ready', weight_id))
        self.sequence += 1
        if self.cancelled is not None and self.cancelled.is_set():
            raise InterruptedError('Encrypted experiment cancelled')
        return result

    def matrix_multiplication(self, encrypted_matrix, data, compression_key):
        # data is the upstream's public quantized weight matrix. Input scales/zero points stay local.
        raw_weights = np.ascontiguousarray(data, dtype='<u8')
        weight_id = digest(raw_weights.tobytes(order='C'))
        if weight_id not in self.matrices or compression_key is not self.key:
            raise ValueError('Unregistered model or key context')
        ciphertext = encrypted_matrix.serialize()
        started = time.perf_counter()
        result = self.exchange('evaluate', ciphertext, weight_id)
        self.remote_seconds += time.perf_counter() - started
        self.calls += 1
        if self.mutate:
            result = self.mutate(result)
        # Full deterministic public replay. No secret key and no attacker ciphertext parser here.
        started = time.perf_counter()
        expected = ext.cpu_matrix_multiplication(encrypted_matrix, self.matrices[weight_id], self.key).serialize()
        self.replay_seconds += time.perf_counter() - started
        if not hmac.compare_digest(expected, result):
            raise ValueError('Encrypted transformer replay rejected output before decryption')
        return ext.CompressedResultEncryptedMatrix.deserialize(result)

    def finish(self):
        self.sent_bytes += send(self.child.stdin, self.profile.frame(self.model_id, self.session, self.key_id, self.sequence, 'finish'), cancelled=self.cancelled)
        self.child.stdin.close()
        if self.child.wait(timeout=POLICY['timeoutSeconds']) != 0 or self.child.stdout.read(1):
            raise ValueError('Encrypted evaluator did not finish cleanly')

    def close(self):
        if self.child.poll() is None:
            self.child.kill()
        self.child.wait()
        for stream in (self.child.stdin, self.child.stdout):
            if stream and not stream.closed:
                stream.close()
        self.key_bytes = b''
