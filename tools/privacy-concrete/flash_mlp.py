"""Complete actual-weight first-layer MLP encrypted route; not full Flash generation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch
import concrete_ml_extensions as ext
from flash_component_plan import write_component,component_parameters
from flash_hyper import FlashHyperModel
from flash_moe import StreamedMoE
from flash_sparse_reference import source_mlp_reference
from flash_mlp_plan import CONFIG,WEIGHTS,SOURCE_ID,SEMANTICS,OPERATIONS,load_plan,versions
from full_bridge import FullBridge
from full_status import ExecutionRejected,failure_event


class StreamedFlashMLP:
    def __init__(self,source_dir,cache_dir):
        self.moe=StreamedMoE(source_dir,cache_dir)
        self.hyper=FlashHyperModel(source_dir,cache_dir)
        self.directory=Path(cache_dir)

    def prepare(self):
        moe=self.moe.prepare();hyper=self.hyper.prepare()
        if component_parameters(CONFIG)!=hyper.definition()['parameters']:raise ValueError('MLP precision profile changed')
        self.moe.executor.glwe_crypto_params=self.hyper.executor.glwe_crypto_params
        definitions=hyper.definition()['projections']+moe.definition()['projections']
        if [row['name'] for row in definitions]!=[row['name'] for row in OPERATIONS]:raise ValueError('Complete MLP route order changed')
        write_component(self.directory,CONFIG,definitions,SEMANTICS,versions(),source=WEIGHTS)
        return load_plan(self.directory)

    def evaluate(self,value,mode,plan,bridge=None,secret=None):
        definitions=plan.definition()['projections']
        prefix=[projection.definition for projection in self.hyper.projections]
        if definitions[:len(prefix)]!=prefix:raise ValueError('Complete MLP hyperconnection route changed')
        def block(mixed):
            result=self.moe.evaluate(mixed.reshape(-1).tolist(),mode,plan,bridge,secret,
                                     projection_definitions=definitions[len(prefix):])
            return result.reshape_as(mixed)
        return self.hyper.evaluate(value,mode,bridge,secret,block=block)[0]


def run(source_dir,cache_dir,execution,reference_path):
    if execution not in ['public-reference','encrypted']:raise ValueError('Explicit Flash MLP execution mode required')
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit MLP CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);began=time.perf_counter()
    model=StreamedFlashMLP(source_dir,cache_dir);plan=model.prepare()
    width=model.hyper.graph.program.streams*model.hyper.graph.program.hidden_width
    value=torch.sin(torch.arange(width,dtype=torch.float32)*.013).reshape(1,-1)*.1
    official=source_mlp_reference(model.moe.source,model.hyper.source,reference_path)(value)
    floating=model.evaluate(value,'float-reference',plan);quantized=model.evaluate(value,'clear-reference',plan)
    if not torch.allclose(floating,official,rtol=1e-5,atol=1e-5):raise ExecutionRejected('component-reference-mismatch')
    report=dict(schema='doppler.actual-flash-mlp-block-execution/v1',fixtureId='flash-first-mlp-block-sine',
                scope='First-layer MLP hyperconnection, shared/router/all-512-expert MoE and residual; excludes attention and Flash generation',
                productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],planId=plan.identity,modelId=plan.definition()['modelId'],
                protocol=plan.protocol_id,linearModules=len(OPERATIONS),plannedEncryptedTiles=len(plan.schedule()),
                allExpertsVisited=True,officialSparseReferenceCheckPassed=True,maxFloatingVsOfficialSparseReferenceError=float((floating-official).abs().max()),
                encryptedExecutionCompleted=False,encryptedTiles=0,execution=execution,
                maxQuantizedVsFloatingError=float((quantized-floating).abs().max()))
    if execution=='public-reference':return dict(report,elapsedMs=(time.perf_counter()-began)*1000)
    secret,key=ext.cpu_create_private_key(model.hyper.executor.glwe_crypto_params);bridge=None
    try:
        bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'])
        started=time.perf_counter();actual=model.evaluate(value,'execute',plan,bridge,secret)
        if not torch.allclose(actual,quantized,rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):
            raise ExecutionRejected('component-reference-mismatch')
        bridge.finish()
        return dict(report,encryptedExecutionCompleted=True,encryptedTiles=bridge.sequence-1,
                    maxEncryptedVsQuantizedError=float((actual-quantized).abs().max()),quantizedReferenceCheckPassed=True,
                    executeMs=(time.perf_counter()-started)*1000,elapsedMs=(time.perf_counter()-began)*1000,
                    evaluationKeyBytes=bridge.key_bytes,sentBytes=bridge.sent,receivedBytes=bridge.received,
                    clientEncryptedReplayMs=bridge.replay_seconds*1000,evaluatorRoundTripMs=bridge.remote_seconds*1000)
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--execution',required=True,choices=['public-reference','encrypted']);parser.add_argument('--reference-source',required=True)
    args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir,args.execution,args.reference_source)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-mlp-block-execution/v1',productionBackendAvailable=False,
                              securityQualified=False,fullFlashModelExecuted=False,
                              status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True)
        raise
    print(json.dumps(report,indent=2))
