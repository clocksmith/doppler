"""Local-process full Flash hybrid research runner; never downloads or provisions.

Use only in an agreed CPU/IO window with the existing public checkpoint and exact
native build. A new cooperative pause request cancels this attempt; it cannot
resume or downgrade a partly evaluated route. Production admission stays closed.
"""
import argparse
from contextlib import contextmanager,ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time
from protocol import require_active
from full_plan import canonical
from full_status import failure_event
from plan_backends import REBUILD_PROFILES,compiled_backend

RESOURCE_POLICY=json.loads(Path(__file__).with_name('flash-resource-policy.json').read_text())


def require_private_resource_scope():
    """Kernel evidence, never a caller-supplied 'budget enforced' flag."""
    try:
        if RESOURCE_POLICY['requireExclusiveLeaf'] is not True or RESOURCE_POLICY['requireGroupOomKill'] is not True:
            raise ValueError()
        rows=[line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
        if len(rows)!=1:raise ValueError()
        path=Path(rows[0])
        if not path.is_absolute() or len(path.parts)<2 or '..' in path.parts:raise ValueError()
        leaf=Path('/sys/fs/cgroup').joinpath(*path.parts[1:])
        if (leaf/'cgroup.procs').read_text().split()!=[str(os.getpid())]:raise ValueError()
        memory=int((leaf/'memory.max').read_text());swap=int((leaf/'memory.swap.max').read_text())
        oom=int((leaf/'memory.oom.group').read_text());quota,period=map(int,(leaf/'cpu.max').read_text().split())
        if (not 0<memory<=RESOURCE_POLICY['maximumMemoryBytes'] or swap!=RESOURCE_POLICY['maximumSwapBytes'] or oom!=1
                or quota<=0 or period<=0 or quota>period*RESOURCE_POLICY['maximumCpuCores']):raise ValueError()
    except (OSError,ValueError):
        raise ValueError('Dedicated enforced CPU/memory/no-swap/group-OOM cgroup required; no resource fallback') from None
    return dict(memoryMaxBytes=memory,swapMaxBytes=swap,cpuQuotaMicros=quota,cpuPeriodMicros=period,groupOomKill=True,
        policyId=hashlib.sha256(canonical(RESOURCE_POLICY)).hexdigest())

class PauseCancellation:
    def __init__(self,pause,cancelled):self.pause,self.cancelled=pause,cancelled
    def is_set(self):return self.pause.exists() or self.cancelled is not None and self.cancelled.is_set()


@contextmanager
def resource_turn(lock_path,pause_path,wait_seconds,*,cancelled=None):
    """Existing shared lock only; never create/remove another owner's pause."""
    lock,pause=Path(lock_path),Path(pause_path)
    if (not lock.is_absolute() or lock.is_symlink() or not lock.is_file() or not pause.is_absolute()
            or type(wait_seconds) not in (int,float) or not 0<wait_seconds<=7200):
        raise ValueError('Existing agreed resource lock, pause path and bounded wait required')
    deadline=time.monotonic()+wait_seconds;acquired=False
    with lock.open('r+') as handle:
        try:
            while not acquired:
                require_active(cancelled)
                if not pause.exists():
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                    except BlockingIOError:pass
                if not acquired:
                    if time.monotonic()>=deadline:raise TimeoutError('No agreed Flash resource turn')
                    time.sleep(.05)
            signal=PauseCancellation(pause,cancelled);require_active(signal)
            yield signal
        finally:
            if acquired:fcntl.flock(handle,fcntl.LOCK_UN)


def run(source_directory,reference_path,scratch_directory,prompt,*,backend_profile,tenant_id,privacy_id,
        lock_path,pause_path,resource_wait_seconds,cancelled=None):
    require_active(cancelled)
    from flash_request_state import ClientContext
    if any(not isinstance(value,str) or not 0<len(value)<=256 for value in (tenant_id,privacy_id)):
        raise ValueError('Explicit bounded tenant and privacy namespace required')
    backend=compiled_backend(backend_profile)
    if backend_profile not in REBUILD_PROFILES:raise ValueError('Explicit installed reconstruction generation profile required')
    settings=backend.CONFIG
    if any(os.environ.get(name)!=str(settings['cpuThreads']) for name in ('OMP_NUM_THREADS','RAYON_NUM_THREADS')):
        raise ValueError('The research process must declare its installed CPU thread budget before native initialization')
    source,scratch,reference=Path(source_directory),Path(scratch_directory),Path(reference_path)
    from flash_source_scopes import TEXT
    from flash_text_plan import GEOMETRY
    largest=max(width*height*2 for width,height in GEOMETRY.values())
    began=time.perf_counter()
    with resource_turn(lock_path,pause_path,resource_wait_seconds,cancelled=cancelled) as active:
        resource_limits=require_private_resource_scope()
        if not source.is_dir() or not scratch.is_dir() or not reference.is_file():raise ValueError('Existing checkpoint, inspected operator source and scratch required')
        immutable={source.resolve(),*((source/name).resolve().parent for name in TEXT['shards'])}
        if any(scratch.resolve().is_relative_to(root) for root in immutable):raise ValueError('Scratch must be outside immutable checkpoint locations')
        if shutil.disk_usage(scratch).free<2*largest+64*1024**2:raise ValueError('Insufficient client and provider temporary coefficient capacity')
        versions=backend.versions() # Exact native/dependencies before source reads or keys.
        from flash_reference import SOURCE as REFERENCE
        for name in ('modeling_qwen4_exp.py','configuration_qwen4_exp.py'):
            if hashlib.sha256((reference.parent/name).read_bytes()).hexdigest()!=REFERENCE['files'][name]['sha256']:
                raise ValueError('Existing inspected Flash reference source required')
        if reference.name!='modeling_qwen4_exp.py':raise ValueError('Explicit inspected Flash modeling file required')
        import torch
        import concrete_ml_extensions as ext
        if not callable(getattr(ext,'cpu_create_private_key',None)):
            raise ValueError('Installed CPU-only native key-generation interface required')
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from flash_tokenizer import encode,decode
        from flash_graph_source import bind_graph
        from flash_stage_source import FlashStageSource
        from local_token_rows import LocalTokenRows
        from local_ngram_rows import LocalNgramRows
        from flash_prepared_graph import PreparedFlashGraph,EPHEMERAL_STORAGE
        from flash_generation import generate
        from flash_float_reference import BoundedFloatingReference
        from checked_bridge import CheckedBridge
        from research_parameters import component_parameters
        previous_threads=torch.get_num_threads();torch.set_num_threads(settings['cpuThreads'])
        bridge=None;secret=key=None
        try:
            tokens=encode(source,prompt,settings['contextBucket'],cancelled=active)
            if len(tokens)!=settings['contextBucket']:raise ValueError('Prompt must exactly match the explicit research prefill profile')
            with ExitStack() as owned:
                temporary=owned.enter_context(tempfile.TemporaryDirectory(dir=scratch,prefix='flash-generation-'))
                decoders=[];verified=0
                for index in range(TEXT['layerCount']):
                    stage=FlashStageSource(source,'decoder',index,cancelled=active)
                    owned.callback(stage.close);decoders.append(stage);verified+=stage.spec['wholeShardBytes']
                output=FlashStageSource(source,'output',cancelled=active);owned.callback(output.close);verified+=output.spec['wholeShardBytes']
                token_rows=LocalTokenRows.flash(source,local_storage_limit_bytes=4*1024**3,cancelled=active);owned.callback(token_rows.close)
                ple_rows=LocalNgramRows.flash(source,local_storage_limit_bytes=128*1024**3,cancelled=active);owned.callback(ple_rows.close)
                verified+=token_rows.spec['wholeShardBytes']+ple_rows.spec['wholeShardBytes']
                graph=bind_graph(decoders,output,token_rows,ple_rows,reference,context_limit=settings['contextBucket']+settings['steps']-1,
                    max_lookup_rows=settings['contextBucket'],max_ple_lookup_bytes=64*1024**2,cancelled=active)
                parameters=ext.MatmulCryptoParameters.deserialize(json.dumps(component_parameters(settings)['parameters']))
                executor=GLWELinearLayerExecutor(use_dynamic_quantization=True);executor.glwe_crypto_params=parameters
                prepared=PreparedFlashGraph(graph,executor,temporary,quantization_bits=12,storage=EPHEMERAL_STORAGE,cancelled=active)
                plan,binding=backend.write_plan(temporary,prepared,cancelled=active)
                floating=BoundedFloatingReference(graph,cancelled=active);owned.callback(floating.close)
                require_active(active);secret,key=ext.cpu_create_private_key(parameters);require_active(active)
                bridge=CheckedBridge(temporary,plan,key,cancelled=active,backend_profile=backend_profile,prepared_public_tiles=binding,
                    parameter_profile=settings['parameterProfile'],provider_source_directory=source)
                owned.callback(bridge.close) # Stop/wait provider before source or scratch cleanup.
                context=ClientContext(tenant_id,privacy_id,graph.identity,bridge.key_id,bridge.session)
                result=generate(prepared,list(tokens),bridge,secret,parameters,context,floating_reference=floating)
                if result['publicCoefficientBytesSent']!=0:raise ValueError('Reconstruction route unexpectedly sent public coefficients')
                result.update(schema='doppler.actual-flash-hybrid-research-execution/v1',backendProfile=backend_profile,
                    transport='local-process',actualFlashCheckpointExecuted=True,encryptedExecution=True,
                    sourceId=hashlib.sha256(canonical(TEXT)).hexdigest(),sourceRevision=TEXT['revision'],
                    linearProjectionCount=len(prepared.order),encryptedTileCount=len(plan.schedule())*settings['steps'],
                    clientLocalOutputText=decode(source,result['clientLocalOutputTokens'],settings['steps'],cancelled=active),
                    implementationVersions=versions,sha256VerifiedBytesIncludingRepeats=verified,
                    resourceLimits=resource_limits,floatingPeakFp32WeightChunkBytes=floating.peak_fp32_weight_bytes,
                    preparationSeconds=prepared.preparation_seconds,rematerializationSeconds=prepared.rematerialization_seconds,
                    elapsedSeconds=time.perf_counter()-began,clientLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                    integrity=bridge.integrity_report(),driverSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                return result
        finally:
            secret=key=None;torch.set_num_threads(previous_threads)


def main(arguments=None):
    from flash_tokenizer import SOURCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--reference',required=True);parser.add_argument('--scratch',required=True)
    parser.add_argument('--backend-profile',required=True,choices=REBUILD_PROFILES)
    parser.add_argument('--tenant-id',required=True);parser.add_argument('--privacy-id',required=True)
    parser.add_argument('--lock',required=True);parser.add_argument('--pause-file',required=True);parser.add_argument('--resource-wait-seconds',required=True,type=int)
    parser.add_argument('--public-fixture',choices=[SOURCE['publicFixture']['id']])
    args=parser.parse_args(arguments);began=time.perf_counter()
    try:
        prompt=SOURCE['publicFixture']['text'] if args.public_fixture else sys.stdin.buffer.read(SOURCE['maxPromptBytes']+1).decode('utf-8')
        result=run(args.source,args.reference,args.scratch,prompt,backend_profile=args.backend_profile,tenant_id=args.tenant_id,privacy_id=args.privacy_id,
            lock_path=args.lock,pause_path=args.pause_file,resource_wait_seconds=args.resource_wait_seconds)
        if args.public_fixture:result['publicFixtureId']=args.public_fixture
        print(json.dumps(result,indent=2));return 0
    except BaseException as error:
        print(json.dumps(failure_event(error,(time.perf_counter()-began)*1000)),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':sys.exit(main())
