"""Real native FHE over all installed tile shapes in separate owned processes.

Synthetic public matrices and feedback only. The committed metadata reservation
is a partial memory scenario, not the full Flash graph or a model-latency result.
Run only in a fresh enforced eight-GiB cgroup; no model downloads or host setup.
"""
import argparse
from contextlib import ExitStack,contextmanager
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from full_plan import canonical,tiles
from protocol import digest

ROOT=Path(__file__).resolve().parent


def run():
    from flash_generate_run import require_private_resource_scope
    limits=require_private_resource_scope() # Kernel evidence before keys/allocation.
    import numpy as np
    import torch
    import concrete_ml_extensions as ext
    from native_process_fixture import fixture
    from flash_component_plan import write_component
    from public_weights import PreparedPublicTiles
    from checked_bridge import CheckedBridge
    from linear_check import honest_norm_bound
    from resource_memory import ResourceMeter
    from flash_compute_budget import project_native_cost
    began=time.perf_counter()
    with ExitStack() as owned:
        root=Path(owned.enter_context(tempfile.TemporaryDirectory(prefix='native-process-probe-')))
        scope=owned.enter_context(fixture(root/'client-public-source'));stream=scope.stream
        plan_root=root/'plan';plan_root.mkdir();provider_root=root/'provider';provider_root.mkdir(mode=0o700)
        provider_report=root/'provider-report.json'
        versions=stream.versions()
        digests={tiles(row,2048,2048)[0]['id']:row['weightId'] for row in scope.rows}
        write_component(plan_root,stream.CONFIG,scope.rows,stream.semantics('d'*64,digests),versions,source=stream.TEXT)
        plan=stream.load_provider_plan(plan_root)
        @contextmanager
        def coefficients(row,encoding):yield io.BytesIO(scope.matrices[row['name']].astype('<i2').tobytes())
        binding=PreparedPublicTiles(plan,coefficients)
        # Physically touch the entire reservation. These are not real model
        # objects, and other graph/proof/allocator overhead is deliberately not
        # inferred from this lower-bound scenario.
        reserve=torch.empty(scope.inventory['logicalQuantizerMetadataBytes']//4,dtype=torch.float32);reserve.fill_(1.)
        parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(plan.definition()['parameters']['parameters']))
        meter=ResourceMeter();owned.callback(meter.close);meter.mark('probe-start');meter.start()
        key_started=time.perf_counter();secret,key=ext.cpu_create_private_key(parameters);key_seconds=time.perf_counter()-key_started
        meter.mark('after-key-generation')
        class Deployment:
            def bind(self,validated):
                if validated.identity!=plan.identity:raise ValueError('Synthetic deployment route changed')
                return validated
            def spawn(self,validated):
                environment=dict(PATH=os.defpath,PYTHONPATH=str(ROOT),LANG='C.UTF-8',PYTHONNOUSERSITE='1',
                    HF_HOME=str(provider_root/'hf-cache'),RAYON_NUM_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                return subprocess.Popen([sys.executable,str(ROOT/'fixtures/native_process_provider.py'),'--directory',str(plan_root),
                    '--plan-id',validated.identity,'--work',str(provider_root),'--report',str(provider_report)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,env=environment)
        bridge=CheckedBridge(plan_root,plan,key,backend_profile=stream.PROFILE,prepared_public_tiles=binding,
            parameter_profile=stream.CONFIG['parameterProfile'],deployment=Deployment())
        owned.callback(bridge.close);meter.mark('after-provider-readiness')
        values={row['name']:scope.generator.integers(0,4096,size=(5,row['inputWidth']),dtype=np.uint64) for row in scope.rows}
        errors=[];client_cases=[]
        for step in range(4):
            for row in scope.rows:
                matrix=scope.matrices[row['name']];value=values[row['name']];tile=tiles(row,2048,2048)[0]
                started=time.perf_counter();sent,received=bridge.sent,bridge.received
                with bridge.projection(row['name'],row['inputWidth'],row['outputWidth'],value.shape[0]):
                    encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=value)
                    answer=bridge.evaluate(encrypted,matrix.astype(np.uint64),tile)
                    decoded=bridge.decode(answer,tile,value,lambda reply:ext.decrypt_matrix(reply,secret,parameters,row['outputWidth']).astype(np.int64))
                expected=value.astype(np.int64)@matrix;deviation=decoded-expected
                if np.linalg.norm(deviation)>honest_norm_bound(row['outputWidth'],bridge.check.noise,bridge.check.policy):
                    raise ValueError('Synthetic native integer error exceeds the installed observation bound')
                if not np.array_equal(decoded.argmax(axis=1),expected.argmax(axis=1)):raise ValueError('Synthetic native greedy control differs')
                errors.append(int(np.abs(deviation).max()))
                client_cases.append(dict(step=step,inputWidth=row['inputWidth'],outputWidth=row['outputWidth'],rows=value.shape[0],
                    roundTripAndClientSeconds=time.perf_counter()-started,sentFrameBytes=bridge.sent-sent,receivedFrameBytes=bridge.received-received))
                token=int(decoded[-1].argmax())%row['inputWidth'];next_value=np.roll(value[-1:],1,axis=1).copy()
                next_value[0,token]=(next_value[0,token]+1)%4096;values[row['name']]=next_value
        bridge.finish();bridge.close();bridge.require_integrity();meter.mark('probe-complete');meter.close()
        provider=json.loads(provider_report.read_text());projection=project_native_cost(provider['cases'])
        if bridge.public_coefficient_bytes_sent!=0 or float(reserve[0])!=1.:raise ValueError('Synthetic resource scenario changed')
        return dict(schema='doppler.native-separate-process-resource-probe/v1',installedWorkload=scope.inventory,
            committedQuantizerMetadataReservationBytes=reserve.numel()*reserve.element_size(),
            resourceScenario='Synthetic matrices plus the full logical Flash quantizer metadata size; excludes actual graph objects, model state and full-model mappings',
            resourceLimits=limits,resourceObservation=meter.report(),provider=provider,clientCases=client_cases,
            weightedSyntheticKernelWork=projection,keyGenerationSeconds=key_seconds,evaluationKeyBytes=bridge.key_bytes,
            elapsedSeconds=time.perf_counter()-began,processPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
            maxIntegerErrors=errors,allGreedyControlsMatch=True,totalFrameBytesSent=bridge.sent,totalFrameBytesReceived=bridge.received,
            publicCoefficientBytesSent=bridge.public_coefficient_bytes_sent,clientSampledReplaySeconds=bridge.replay_seconds,
            clientCheckSeconds=bridge.check_seconds,integrity=bridge.integrity_report(),implementationVersions=versions,
            separateProcesses=True,separateMachine=False,nativeEncryptionSimulated=False,syntheticGeometryFixture=True,
            actualFlashCheckpointExecuted=False,fullModelLatencyMeasured=False,productionBackendAvailable=False,securityQualified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run()
    with Path(args.output).open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
