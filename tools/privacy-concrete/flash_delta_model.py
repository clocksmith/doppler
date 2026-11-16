"""Actual first-layer Flash linear-attention encrypted projections over four cached steps."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import torch
from torch import nn
import concrete_ml_extensions as ext
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
from full_model import FullLinear
from full_bridge import FullBridge
from full_status import ExecutionRejected,failure_event
from flash_component_plan import write_component
from flash_delta import ClientDeltaNet
from flash_delta_state import DeltaContext,DeltaRequestState
from flash_delta_plan import CONFIG,WEIGHTS,SOURCE_ID,PREFIX,ORDER,PROGRAM_ID,load_plan,versions
from flash_weights import FlashSource
from protocol import POLICY


class ActualFlashDelta:
    def __init__(self,source_dir,cache_dir,reference_path):
        self.source=FlashSource(source_dir,scope='linear-attention')
        self.directory=Path(cache_dir);self.directory.mkdir(parents=True,exist_ok=True)
        config=SimpleNamespace(**self.source.config['text_config'])
        # Meta construction avoids random initialization and duplicate public weight allocations.
        with torch.device('meta'):self.graph=ClientDeltaNet(reference_path,config)
        for name in WEIGHTS['tensors']:
            leaf=name[len(PREFIX):];parent,_,field=leaf.rpartition('.')
            module=self.graph.model.get_submodule(parent) if parent else self.graph.model
            if parent in ORDER:module=module.source
            setattr(module,field,nn.Parameter(self.source.tensor(name),requires_grad=False))
        self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
        self.projections={name:FullLinear(name,getattr(self.graph.model,name).source,self.executor,self.directory) for name in ORDER}

    def prepare(self):
        installed=versions()
        if json.loads(ext.default_params())!=POLICY['parameters']:raise ValueError('Pinned native parameters required')
        definitions=[self.projections[name].definition for name in ORDER]
        write_component(self.directory,CONFIG,definitions,dict(programId=PROGRAM_ID),installed,source=WEIGHTS)
        return load_plan(self.directory)

    def evaluate(self,value,state,context,mode,bridge=None,secret=None):
        if mode=='float-reference':return self.graph(value,state,context,execution='public-reference')
        if mode not in ['clear-reference','execute']:raise ValueError('Explicit actual DeltaNet execution mode required')
        def project(operation,value):
            projection=self.projections[operation['id']]
            if (operation['inputWidth'],operation['outputWidth'])!=(projection.definition['inputWidth'],projection.definition['outputWidth']):
                raise ValueError('Actual DeltaNet projection geometry changed')
            return projection(value)
        for projection in self.projections.values():
            projection.mode=mode;projection.bridge=bridge;projection.secret=secret;projection.parameters=self.executor.glwe_crypto_params
        try:return self.graph(value,state,context,execution='verified-projections',verified_client_projection=project)
        finally:
            for projection in self.projections.values():projection.secret=projection.bridge=projection.parameters=None


def run(source_dir,cache_dir,reference_path):
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit actual DeltaNet CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);began=time.perf_counter()
    model=ActualFlashDelta(source_dir,cache_dir,reference_path);plan=model.prepare()
    width=model.graph.hidden_width
    # Public layer inputs, not a private prompt or complete-model token trajectory.
    inputs=torch.sin(torch.arange(CONFIG['steps']*width,dtype=torch.float32)*.013).reshape(CONFIG['steps'],1,1,width)*.1
    references={}
    for mode in ['float-reference','clear-reference']:
        context=DeltaContext('public-fixture',plan.identity,plan.definition()['modelId'],mode,mode)
        state=DeltaRequestState(context)
        try:references[mode]=[model.evaluate(value,state,context,mode) for value in inputs]
        finally:state.close()
    secret,key=ext.cpu_create_private_key(model.executor.glwe_crypto_params);bridge=None;state=None
    try:
        bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'],
                          observe=lambda event,value:print(json.dumps(dict(event=event,**value)),file=sys.stderr,flush=True))
        context=DeltaContext('public-fixture',plan.identity,plan.definition()['modelId'],bridge.key_id,bridge.session)
        state=DeltaRequestState(context);errors=[];step_ms=[]
        for index,value in enumerate(inputs):
            started=time.perf_counter();actual=model.evaluate(value,state,context,'execute',bridge,secret)
            if not torch.allclose(actual,references['clear-reference'][index],rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):
                raise ExecutionRejected('component-reference-mismatch')
            errors.append(float((actual-references['clear-reference'][index]).abs().max()));step_ms.append((time.perf_counter()-started)*1000)
        bridge.finish()
        return dict(schema='doppler.actual-flash-delta-execution/v1',fixtureId='flash-first-delta-four-public-inputs',
                    scope='Four actual-weight cached first-layer DeltaNet steps, encrypted linear projections; client convolution, recurrence and nonlinear operations',
                    productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                    sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],programId=PROGRAM_ID,
                    modelId=plan.definition()['modelId'],planId=plan.identity,protocol=plan.protocol_id,
                    componentSteps=CONFIG['steps'],linearModules=len(ORDER),encryptedTiles=bridge.sequence-1,
                    quantizedReferenceCheckPassed=True,maxEncryptedVsQuantizedErrors=errors,
                    maxQuantizedVsFloatingErrors=[float((left-right).abs().max()) for left,right in zip(references['clear-reference'],references['float-reference'])],
                    stepMs=step_ms,elapsedMs=(time.perf_counter()-began)*1000,evaluationKeyBytes=bridge.key_bytes,
                    sentBytes=bridge.sent,receivedBytes=bridge.received,clientEncryptedReplayMs=bridge.replay_seconds*1000,
                    evaluatorRoundTripMs=bridge.remote_seconds*1000)
    finally:
        try:
            if state is not None:state.close()
            if bridge is not None:bridge.close()
        finally:secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--reference-source',required=True);args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir,args.reference_source)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-delta-execution/v1',productionBackendAvailable=False,
                              securityQualified=False,fullFlashModelExecuted=False,
                              status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True)
        raise
    print(json.dumps(report,indent=2))
