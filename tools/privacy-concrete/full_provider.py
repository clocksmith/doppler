"""All-linear evaluator. Receives only evaluation material and ciphertexts; reads public int8 weights."""
import sys
import argparse
import numpy as np
import concrete_ml_extensions as ext
from protocol import check, digest, receive, send


def serve():
    from plan_backends import PROFILES, load_backend_plan
    parser=argparse.ArgumentParser()
    parser.add_argument('directory');parser.add_argument('plan_id')
    parser.add_argument('--profile',choices=PROFILES,default='qwen-all-linear')
    parser.add_argument('--ssh-route',nargs=2)
    args=parser.parse_args()
    directory,plan_id=args.directory,args.plan_id
    plan=load_backend_plan(directory,args.profile)
    if plan.identity!=plan_id:raise ValueError('Installed plan mismatch')
    if args.ssh_route:
        from ssh_route import SSHRoute
        route=SSHRoute.from_file(args.ssh_route[0])
        if route.identity!=args.ssh_route[1]:raise ValueError('SSH route identity changed')
        plan=route.bind(plan,(route.definition()['operatorId'],))
    timeout=plan.definition()['operationTimeoutSeconds']
    source, target = sys.stdin.buffer.raw, sys.stdout.buffer.raw
    header, data, _ = receive(source,timeout=timeout)
    session, key_id = header['session'], header['keyId']
    check(header, plan.frame(session, key_id, 0, 'evaluation-key'))
    if digest(data) != key_id or len(session) != 32:
        raise ValueError('Invalid key context')
    key = ext.CpuCompressionKey.deserialize(data)
    send(target, plan.frame(session, key_id, 0, 'ready'),timeout=timeout)
    weights = {row['name']: np.memmap(f"{directory}/{row['weightId']}.i8", dtype=np.int8, mode='r',
                                     shape=(row['inputWidth'],row['outputWidth'])) for row in plan.definition()['projections']}
    sequence = 1
    for _step in range(plan.definition()['steps']):
        for tile in plan.schedule():
            header, data, _ = receive(source,timeout=timeout)
            check(header, plan.frame(session, key_id, sequence, 'evaluate', tile['id']))
            encrypted = ext.EncryptedMatrix.deserialize(data)
            matrix = np.ascontiguousarray(weights[tile['name']][tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']], dtype=np.int64).astype(np.uint64)
            result = ext.cpu_matrix_multiplication(encrypted, matrix, key)
            send(target, plan.frame(session, key_id, sequence, 'result', tile['id']), result.serialize(),timeout=timeout)
            sequence += 1
    header, data, _ = receive(source,timeout=timeout)
    check(header, plan.frame(session, key_id, sequence, 'finish'))
    if data:
        raise ValueError('Unexpected final payload')


if __name__ == '__main__':
    try:
        serve()
    except BaseException:
        sys.stderr.write('Full encrypted evaluator rejected a frame.\n')
        sys.exit(1)
