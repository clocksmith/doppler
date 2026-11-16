"""Actual pinned Flash expert-0 encrypted SwiGLU; fixed public fixture, not a full Flash backend."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import torch
from torch import nn
import concrete_ml_extensions as ext
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
from full_model import FullLinear
from full_bridge import FullBridge
from verified_execution import complete_verified_route,VALIDATION_BOUNDARY
from flash_component_plan import write_component
from flash_expert_plan import POLICY_EXPERT as CONFIG, SOURCE_ID, WEIGHTS, load_plan, versions
from flash_weights import FlashSource
from full_status import ExecutionRejected, failure_event
from protocol import POLICY

PREFIX='model.language_model.layers.0.mlp.experts.'


def prepare(source_dir,cache_dir):
    installed=versions()
    if json.loads(ext.default_params())!=POLICY['parameters']:raise ValueError('Pinned native parameters required')
    source=FlashSource(source_dir)
    directory=Path(cache_dir);directory.mkdir(parents=True,exist_ok=True)
    executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
    projections=[]
    for leaf in ['gate_up','down']:
        weight=source.tensor(PREFIX+leaf+'_proj',expert=CONFIG['expertIndex'])
        with torch.device('meta'):linear=nn.Linear(weight.shape[1],weight.shape[0],bias=False)
        linear.weight=nn.Parameter(weight,requires_grad=False)
        projections.append(FullLinear('expert/0/'+leaf,linear,executor,directory))
    definitions=[projection.definition for projection in projections]
    write_component(directory,CONFIG,definitions,dict(expertIndex=CONFIG['expertIndex']),installed)
    return projections,executor,load_plan(directory)


def evaluate(projections,hidden,mode):
    for projection in projections:projection.mode=mode
    gate,up=projections[0](hidden).chunk(2,dim=-1)
    return projections[1](torch.nn.functional.silu(gate)*up)


def run(source_dir,cache_dir):
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit component CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);started=time.perf_counter()
    projections,executor,plan=prepare(source_dir,cache_dir)
    # This entry point accepts only an explicit public, full-width component fixture.
    # No stdin/private prompt or input-dependent expert selection is supported.
    hidden=torch.sin(torch.arange(2560,dtype=torch.float32)*.013).reshape(1,2560)
    with torch.no_grad():
        floating=evaluate(projections,hidden,'float-reference')
        quantized=evaluate(projections,hidden,'clear-reference')
    secret,key=ext.cpu_create_private_key(executor.glwe_crypto_params);bridge=None
    try:
        bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'],
                          observe=lambda event,value:print(json.dumps(dict(event=event,**value)),file=sys.stderr,flush=True))
        for projection in projections:
            projection.bridge,projection.secret,projection.parameters=bridge,secret,executor.glwe_crypto_params
        began=time.perf_counter()
        def validate(step,actual):
            if not torch.allclose(actual,quantized,rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):
                raise ExecutionRejected('component-reference-mismatch')
        with torch.no_grad():encrypted=complete_verified_route(bridge,1,lambda step:evaluate(projections,hidden,'execute'),validate)[0]
        execute_ms=(time.perf_counter()-began)*1000
        def rss(kind):
            value=resource.getrusage(kind).ru_maxrss
            return value if platform.system()=='Darwin' else value*1024
        return dict(schema='doppler.actual-flash-expert-execution/v1',fixtureId='flash-expert0-sine-2560',
                    validationBoundary=VALIDATION_BOUNDARY,scope='First-layer fixed public expert 0, complete gated/up/down component; client SiLU and full encrypted replay',
                    productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                    sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],modelId=plan.definition()['modelId'],
                    planId=plan.identity,executionRouteId=bridge.plan.identity,protocol=plan.protocol_id,
                    verifiedSourceBytes=sum(item['bytes'] for item in [*WEIGHTS['files'].values(),*WEIGHTS['shards'].values()]),
                    linearModules=2,logicalProjections=3,encryptedTiles=bridge.sequence-1,
                    maxEncryptedVsQuantizedError=float((encrypted-quantized).abs().max()),
                    maxQuantizedVsFloatingError=float((quantized-floating).abs().max()),
                    quantizedReferenceCheckPassed=True,outputElements=encrypted.numel(),
                    executeMs=execute_ms,elapsedMs=(time.perf_counter()-started)*1000,
                    clientEncryptedReplayMs=bridge.replay_seconds*1000,evaluatorRoundTripMs=bridge.remote_seconds*1000,
                    evaluationKeyBytes=bridge.key_bytes,sentBytes=bridge.sent,receivedBytes=bridge.received,
                    clientLifetimePeakRssBytes=rss(resource.RUSAGE_SELF),reapedChildrenLifetimePeakRssBytes=rss(resource.RUSAGE_CHILDREN),
                    implementationVersions=versions())
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:
            for projection in projections:projection.secret=projection.bridge=projection.parameters=None
            secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir)
    except BaseException as error:
        # Construct a bounded status at the source. No exception text, paths, key material,
        # raw vectors or native diagnostic output is included in the public report.
        report=dict(schema='doppler.actual-flash-expert-execution/v1',fixtureId='flash-expert0-sine-2560',
                    productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                    status=failure_event(error,(time.perf_counter()-began)*1000))
        print(json.dumps(report,indent=2),flush=True)
        raise
    print(json.dumps(report,indent=2))
