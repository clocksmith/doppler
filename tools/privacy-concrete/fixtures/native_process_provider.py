"""Owned synthetic evaluator process for native resource/cost qualification only."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import resource
import sys
import time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--plan-id',required=True)
    parser.add_argument('--work',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    from native_process_fixture import fixture
    from flash_provider_source import ProviderCheckpointSource
    import full_provider
    began=time.perf_counter();cases=[]
    with ExitStack() as owned:
        work=Path(args.work);scope=owned.enter_context(fixture(work/'source'))
        plan=scope.stream.load_provider_plan(args.directory)
        if plan.identity!=args.plan_id:raise ValueError('Synthetic process plan identity changed')
        scratch=work/'coefficients';scratch.mkdir(mode=0o700)
        checkpoint=ProviderCheckpointSource(scope.source_directory,scratch,plan);owned.callback(checkpoint.close)
        evaluate=full_provider.matrix_evaluator(plan);schedule=plan.schedule()
        def measured(encrypted,matrix,key):
            index=len(cases);tile=schedule[index%len(schedule)];step=index//len(schedule)
            started=time.perf_counter();result=evaluate(encrypted,matrix,key)
            cases.append(dict(step=step,rows=5 if step==0 else 1,inputWidth=tile['inputEnd']-tile['inputBegin'],
                outputWidth=tile['outputEnd']-tile['outputBegin'],nativeEvaluateSeconds=time.perf_counter()-started))
            return result
        owned.enter_context(patch.object(full_provider,'matrix_evaluator',return_value=measured))
        # No secret key, plaintext activation, token or decryption feedback is
        # accepted by this process. Only the normal admitted frames are read.
        full_provider.evaluate_route(plan,args.directory,sys.stdin.buffer.raw,sys.stdout.buffer.raw,checkpoint_source=checkpoint)
        if len(cases)!=len(schedule)*4 or list(scratch.iterdir()):raise ValueError('Synthetic fixed route or owned cleanup incomplete')
        report=dict(schema='doppler.native-process-provider-observation/v1',cases=cases,elapsedSeconds=time.perf_counter()-began,
            processPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
            materializations=checkpoint.materializations,peakCoefficientDiskBytes=checkpoint.peak_cache_bytes,
            sourceBytesVerified=checkpoint.source_bytes_verified,sourceVerificationSeconds=checkpoint.source_verification_seconds,
            materializationSeconds=checkpoint.materialization_seconds,coefficientBytesMaterialized=checkpoint.coefficient_bytes_materialized,
            actualFlashCheckpointExecuted=False,productionBackendAvailable=False,securityQualified=False)
    with Path(args.report).open('x') as handle:handle.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        sys.stderr.write('Synthetic native process probe failed.\n');sys.exit(1)
