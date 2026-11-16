"""All-linear evaluator. Receives only evaluation material and ciphertexts; reads public int8 weights."""
import sys
import numpy as np
import concrete_ml_extensions as ext
from full_plan import load_plan
from protocol import check, digest, receive, send


def serve():
    directory, plan_id = sys.argv[1:]
    plan = load_plan(directory)
    if plan.identity != plan_id:
        raise ValueError('Installed plan mismatch')
    source, target = sys.stdin.buffer.raw, sys.stdout.buffer.raw
    header, data, _ = receive(source)
    session, key_id = header['session'], header['keyId']
    check(header, plan.frame(session, key_id, 0, 'evaluation-key'))
    if digest(data) != key_id or len(session) != 32:
        raise ValueError('Invalid key context')
    key = ext.CpuCompressionKey.deserialize(data)
    send(target, plan.frame(session, key_id, 0, 'ready'))
    weights = {row['name']: np.memmap(f"{directory}/{row['weightId']}.i8", dtype=np.int8, mode='r',
                                     shape=(row['inputWidth'],row['outputWidth'])) for row in plan.definition()['projections']}
    sequence = 1
    for _step in range(plan.definition()['steps']):
        for tile in plan.schedule():
            header, data, _ = receive(source)
            check(header, plan.frame(session, key_id, sequence, 'evaluate', tile['id']))
            encrypted = ext.EncryptedMatrix.deserialize(data)
            matrix = np.ascontiguousarray(weights[tile['name']][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']], dtype=np.int64).astype(np.uint64)
            result = ext.cpu_matrix_multiplication(encrypted, matrix, key)
            send(target, plan.frame(session, key_id, sequence, 'result', tile['id']), result.serialize())
            sequence += 1
    header, data, _ = receive(source)
    check(header, plan.frame(session, key_id, sequence, 'finish'))
    if data:
        raise ValueError('Unexpected final payload')


if __name__ == '__main__':
    try:
        serve()
    except BaseException:
        sys.stderr.write('Full encrypted evaluator rejected a frame.\n')
        sys.exit(1)
