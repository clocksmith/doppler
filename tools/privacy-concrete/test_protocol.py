"""Dependency-free hostile framing and immutable-binding checks."""
import io
import json
import os
import struct
import unittest
import threading
from protocol import POLICY, binding, check, receive, send


class ProtocolTests(unittest.TestCase):
    def decode(self, payload):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, payload)
            os.close(write_fd)
            write_fd = None
            with os.fdopen(read_fd, 'rb', buffering=0) as stream:
                return receive(stream, timeout=1)
        finally:
            if write_fd is not None:
                os.close(write_fd)

    def test_exact_frame_roundtrip_without_plaintext_fields(self):
        header = binding('a' * 64, 'b' * 32, 'c' * 64, 9, 'evaluate', 'd' * 64)
        out = io.BytesIO()
        size = send(out, header, b'encrypted-library-data')
        actual, payload, received_size = self.decode(out.getvalue())
        self.assertEqual((actual, payload, received_size), (header, b'encrypted-library-data', size))
        for key in ['token', 'prompt', 'secretKey', 'inputHash', 'scale', 'attentionMask']:
            with self.assertRaises(ValueError):
                send(io.BytesIO(), dict(header, **{key: 'private'}))

    def test_all_bindings_are_checked_before_ciphertext_use(self):
        expected = binding('a' * 64, 'b' * 32, 'c' * 64, 9, 'result', 'd' * 64)
        check(expected, expected)
        for field in expected:
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(dict(expected, **{field: 'changed'}), expected)

    def test_boolean_sequence_does_not_expand_integer_binding(self):
        expected = binding('m', 's', 'k', 1, 'result')
        with self.assertRaises(ValueError):
            check(dict(expected, sequence=True), expected)

    def test_cancelled_reads_stop_without_another_frame(self):
        read_fd, write_fd = os.pipe()
        event = threading.Event()
        timer = threading.Timer(0.02, event.set)
        timer.start()
        try:
            with os.fdopen(read_fd, 'rb', buffering=0) as stream, self.assertRaises(InterruptedError):
                receive(stream, timeout=5, cancelled=event)
        finally:
            timer.cancel()
            os.close(write_fd)

    def test_oversized_and_truncated_frames_reject(self):
        for payload in [struct.pack('!IQ', POLICY['maxHeaderBytes'] + 1, 0),
                        struct.pack('!IQ', 0, POLICY['maxPayloadBytes'] + 1), b'\0',
                        struct.pack('!IQ', 2, 1) + b'{}']:
            with self.subTest(payload=payload[:12]), self.assertRaises(ValueError):
                self.decode(payload)

    def test_key_upload_to_a_nonreading_provider_is_cancellable(self):
        read_fd, write_fd = os.pipe()
        event = threading.Event()
        timer = threading.Timer(0.02, event.set)
        timer.start()
        try:
            with os.fdopen(write_fd, 'wb', buffering=0) as stream, self.assertRaises(InterruptedError):
                send(stream, binding('m', 's', 'k', 0, 'evaluation-key'), b'x' * 1000000,
                     timeout=5, cancelled=event)
                self.fail('blocked key upload returned')
        finally:
            timer.cancel()
            os.close(read_fd)

    def test_duplicate_headers_reject(self):
        header = json.dumps(binding('m', 's', 'k', 0, 'ready'))
        duplicate = (header[:-1] + ',"session":"other"}').encode()
        with self.assertRaises(ValueError):
            self.decode(struct.pack('!IQ', len(duplicate), 0) + duplicate)

    def test_a_missing_response_is_bounded_by_deadline(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, 'rb', buffering=0) as stream, self.assertRaises(TimeoutError):
                receive(stream, timeout=0.01)
        finally:
            os.close(write_fd)


if __name__ == '__main__':
    unittest.main()
