"""Fresh evaluator interpreter. No client keys, tokens, scales, masks or activations are inputs."""
import sys
import numpy as np
import concrete_ml_extensions as ext
from protocol import check, digest, receive, send
from profiles import load_profile


def serve():
    profile = load_profile(sys.argv[2] if len(sys.argv) == 3 else 'tiny')
    source, target = sys.stdin.buffer.raw, sys.stdout.buffer.raw
    header, payload, _ = receive(source)
    model_id = sys.argv[1]
    session, key_id = header['session'], header['keyId']
    if len(session) != 32 or len(key_id) != 64 or digest(payload) != key_id:
        raise ValueError('Invalid public evaluation material')
    check(header, profile.frame(model_id, session, key_id, 0, 'evaluation-key'))
    key = ext.CpuCompressionKey.deserialize(payload)
    send(target, profile.frame(model_id, session, key_id, 0, 'ready'))
    sequence, weights, calls = 1, {}, 0
    while True:
        header, payload, _ = receive(source)
        kind, weight_id = header['kind'], header['weightId']
        check(header, profile.frame(model_id, session, key_id, sequence, kind, weight_id))
        if kind == 'public-weight':
            if len(weights) >= profile.projection_count or len(payload) != profile.input_width * profile.output_width * 8 or digest(payload) != weight_id or weight_id in weights or profile.expected_weight_ids and weight_id not in profile.expected_weight_ids:
                raise ValueError('Invalid public model matrix')
            weights[weight_id] = np.frombuffer(payload, dtype='<u8').reshape(profile.input_width, profile.output_width)
            send(target, profile.frame(model_id, session, key_id, sequence, 'ready', weight_id))
        elif kind == 'evaluate':
            if len(weights) != profile.projection_count or weight_id not in weights or calls >= profile.steps * profile.projection_count:
                raise ValueError('Unregistered public model')
            encrypted = ext.EncryptedMatrix.deserialize(payload)
            result = ext.cpu_matrix_multiplication(encrypted, weights[weight_id], key)
            send(target, profile.frame(model_id, session, key_id, sequence, 'result', weight_id), result.serialize())
            calls += 1
        elif kind == 'finish' and not payload and not weight_id and calls == profile.steps * profile.projection_count:
            return
        else:
            raise ValueError('Unsupported encrypted operation')
        sequence += 1


if __name__ == '__main__':
    try:
        serve()
    except BaseException:
        sys.stderr.write('Encrypted transformer evaluator rejected a frame.\n')
        sys.exit(1)
