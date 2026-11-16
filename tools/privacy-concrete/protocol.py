"""Bounded binary frames. Payloads are library ciphertexts/public evaluation material only."""
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import time

POLICY = json.loads(Path(__file__).with_name('policy.json').read_text())
POLICY_ID = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
FIELDS = {'schema', 'policyId', 'modelId', 'session', 'keyId', 'sequence', 'kind', 'weightId'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(model_id, session, key_id, sequence, kind, weight_id=''):
    return dict(schema=POLICY['protocol'], policyId=POLICY_ID, modelId=model_id,
                session=session, keyId=key_id, sequence=sequence, kind=kind, weightId=weight_id)


def require_active(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError('Encrypted experiment cancelled')


def send(stream, header, payload=b'', timeout=None, cancelled=None):
    encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode()
    if set(header) != FIELDS or len(encoded) > POLICY['maxHeaderBytes'] or len(payload) > POLICY['maxPayloadBytes']:
        raise ValueError('Encrypted protocol frame rejected')
    data = struct.pack('!IQ', len(encoded), len(payload)) + encoded
    deadline = time.monotonic() + (POLICY['timeoutSeconds'] if timeout is None else timeout)
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        descriptor = None
    was_blocking = None
    try:
        if descriptor is not None:
            was_blocking = os.get_blocking(descriptor)
            os.set_blocking(descriptor, False)
        for chunk in (data, payload):
            view = memoryview(chunk)
            while view:
                require_active(cancelled)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Encrypted protocol deadline exceeded')
                if descriptor is None:
                    written = stream.write(view)
                else:
                    interval = min(remaining, 0.05) if cancelled is not None else remaining
                    if not select.select([], [descriptor], [], interval)[1]:
                        continue
                    try:
                        written = os.write(descriptor, view[:65536])
                    except BlockingIOError:
                        continue
                if not written:
                    raise ValueError('Encrypted protocol write failed')
                view = view[written:]
        return len(data) + len(payload)
    finally:
        if was_blocking is not None:
            os.set_blocking(descriptor, was_blocking)


def receive(stream, timeout=None, cancelled=None):
    deadline = time.monotonic() + (POLICY['timeoutSeconds'] if timeout is None else timeout)

    def exact(size):
        data = bytearray()
        while len(data) < size:
            require_active(cancelled)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Encrypted protocol deadline exceeded')
            interval = min(remaining, 0.05) if cancelled is not None else remaining
            if not select.select([stream], [], [], interval)[0]:
                continue
            chunk = os.read(stream.fileno(), min(size - len(data), 1024 * 1024))
            if not chunk:
                raise ValueError('Encrypted protocol ended early')
            data.extend(chunk)
        return bytes(data)

    header_size, payload_size = struct.unpack('!IQ', exact(12))
    if header_size > POLICY['maxHeaderBytes'] or payload_size > POLICY['maxPayloadBytes']:
        raise ValueError('Encrypted protocol frame rejected')

    header = parse_unique_json(exact(header_size))
    if not isinstance(header, dict) or set(header) != FIELDS:
        raise ValueError('Encrypted protocol frame rejected')
    return header, exact(payload_size), 12 + header_size + payload_size



def parse_unique_json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Encrypted protocol duplicate field')
            value[key] = item
        return value
    return json.loads(data, object_pairs_hook=unique)


def check(actual, expected):
    if actual != expected or set(actual) != FIELDS or any(type(actual[key]) is not type(expected[key]) for key in FIELDS):
        raise ValueError('Encrypted protocol binding rejected')
