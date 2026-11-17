"""All-linear evaluator. Receives only evaluation material and ciphertexts; reads the admitted public coefficient encoding."""
import sys
from contextlib import nullcontext,ExitStack
import numpy as np
import concrete_ml_extensions as ext
from protocol import check, digest, receive, send
from weight_encoding import weight_encoding
from public_weights import PublicTileReader
from native_runtime import matrix_evaluator,prepare_evaluator_key


def serve():
    from plan_backends import load_provider_plan, provider_arguments
    args=provider_arguments(sys.argv[1:])
    directory,plan_id=args.directory,args.plan_id
    plan=load_provider_plan(directory,args.profile)
    if plan.identity!=plan_id:raise ValueError('Installed plan mismatch')
    if args.ssh_route:
        from ssh_route import SSHRoute
        route=SSHRoute.from_file(args.ssh_route[0])
        if route.identity!=args.ssh_route[1]:raise ValueError('SSH route identity changed')
        if args.public_source_dir!=route.definition().get('remotePublicSource'):
            raise ValueError('Provider checkpoint path differs from the explicit SSH route')
        plan=route.bind(plan,(route.definition()['operatorId'],))
    from public_tile_stream import REBUILD_CONTRACT
    rebuild=plan.definition().get('publicCoefficientStorage')==REBUILD_CONTRACT
    if (args.public_source_dir is not None)!=rebuild:raise ValueError('Explicit existing provider checkpoint required only by the rebuild profile')
    with ExitStack() as owned:
        checkpoint=None
        if rebuild:
            from flash_provider_source import ProviderCheckpointSource
            checkpoint=ProviderCheckpointSource(args.public_source_dir,directory,plan);owned.callback(checkpoint.close)
        evaluate_route(plan,directory,sys.stdin.buffer.raw,sys.stdout.buffer.raw,checkpoint_source=checkpoint)


def evaluate_route(plan,directory,source,target,*,checkpoint_source=None):
    """Serve one fixed admitted route on owned streams, including bounded storage."""
    from public_tile_stream import CONTRACT,REBUILD_CONTRACT,decode
    definition=plan.definition();storage=definition.get('publicCoefficientStorage')
    if storage not in (None,CONTRACT,REBUILD_CONTRACT):raise ValueError('Unsupported public coefficient storage contract')
    rebuild=storage==REBUILD_CONTRACT;streamed=storage==CONTRACT;compiled=rebuild or streamed
    if rebuild:
        from flash_provider_source import ProviderCheckpointSource
        if type(checkpoint_source) is not ProviderCheckpointSource or checkpoint_source.plan_id!=plan.identity:
            raise ValueError('Whole-verified provider checkpoint for this exact route required')
    elif checkpoint_source is not None:raise ValueError('Provider checkpoint cannot expand a legacy storage route')
    evaluate=matrix_evaluator(plan)
    timeout=definition['operationTimeoutSeconds']
    payload_limit=definition['parameters'].get('maxPayloadBytes')
    header, data, _ = receive(source,timeout=timeout,max_payload_bytes=payload_limit)
    session, key_id = header['session'], header['keyId']
    check(header, plan.frame(session, key_id, 0, 'evaluation-key'))
    if digest(data) != key_id or len(session) != 32:
        raise ValueError('Invalid key context')
    key = ext.CpuCompressionKey.deserialize(data)
    prepare_evaluator_key(plan,key)
    send(target, plan.frame(session, key_id, 0, 'ready'),timeout=timeout,max_payload_bytes=payload_limit)
    encoding=weight_encoding(definition['parameters'])
    sequence = 1
    digests=definition.get('publicTileDigests') if compiled else None
    opener=None if not rebuild else checkpoint_source.open_projection
    lifetime=None if not rebuild else checkpoint_source.require_tile_source
    with (nullcontext(None) if streamed else PublicTileReader(directory,plan,open_projection=opener,check_active=lifetime)) as weights:
        schedule=plan.schedule()
        for _step in range(plan.definition()['steps']):
            for tile in schedule:
                header, data, _ = receive(source,timeout=timeout,max_payload_bytes=payload_limit)
                check(header, plan.frame(session, key_id, sequence, 'evaluate', tile['id']))
                shape=(tile['inputEnd']-tile['inputBegin'],tile['outputEnd']-tile['outputBegin'])
                if streamed:public,data=decode(data,tile,encoding,digests[tile['id']])
                else:public=weights.read(tile)
                coefficients=np.frombuffer(public,dtype=encoding['dtype']).reshape(shape)
                if compiled:
                    if digest(public)!=digests[tile['id']]:raise ValueError('Provider public coefficient tile differs from client preparation')
                    bound=2**(encoding['quantizationBits']-1)
                    if coefficients.min() < -bound or coefficients.max()>=bound:raise ValueError('Public coefficient exceeds declared precision')
                matrix=coefficients.astype(np.int64).astype(np.uint64)
                encrypted = ext.EncryptedMatrix.deserialize(data)
                result = evaluate(encrypted, matrix, key)
                send(target, plan.frame(session, key_id, sequence, 'result', tile['id']), result.serialize(),timeout=timeout,max_payload_bytes=payload_limit)
                sequence += 1
    header, data, _ = receive(source,timeout=timeout,max_payload_bytes=payload_limit)
    check(header, plan.frame(session, key_id, sequence, 'finish'))
    if data:
        raise ValueError('Unexpected final payload')


if __name__ == '__main__':
    try:
        serve()
    except BaseException:
        sys.stderr.write('Full encrypted evaluator rejected a frame.\n')
        sys.exit(1)
