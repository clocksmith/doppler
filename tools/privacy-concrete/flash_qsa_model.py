"""Actual Flash QSA projections with client attention and four cached public-input steps.

This is a component, not Flash token generation. The provider receives encrypted
linear inputs; all attention/indexing, positions, nonlinear operations and replay
are client-owned. No production cryptographic or malicious-security claim.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import resource
import platform
import torch
from torch import nn
import concrete_ml_extensions as ext
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
from full_model import FullLinear
from full_bridge import FullBridge
from verified_execution import complete_verified_route,VALIDATION_BOUNDARY
from full_status import ExecutionRejected,failure_event
from flash_component_plan import write_component
from flash_attention import ClientQSAAttention
from flash_attention_state import AttentionRequestState
from flash_request_state import ClientContext
from flash_qsa_plan import CONFIG,WEIGHTS,SOURCE_ID,PREFIX,ORDER,PROGRAM,PROGRAM_ID,load_plan,versions
from flash_weights import FlashSource
from flash_rotary import text_rotary
from protocol import POLICY


class ActualFlashQSA:
    def __init__(self,source_dir,cache_dir,reference_path):
        self.source=FlashSource(source_dir,scope='full-attention')
        self.directory=Path(cache_dir);self.directory.mkdir(parents=True,exist_ok=True)
        config=SimpleNamespace(**self.source.config['text_config']);config._attn_implementation='eager'
        self.config=config;self.reference_path=reference_path
        rotary=config.rope_parameters
        if rotary['rope_type']!='default':raise ValueError('Only inspected default text rotary is installed')
        width=int(config.head_dim*rotary['partial_rotary_factor'])
        with torch.device('meta'):self.graph=ClientQSAAttention(reference_path,config,context_limit=PROGRAM['contextLimit'],rotary_width=width,layer_index=3)
        for name in WEIGHTS['tensors']:
            leaf=name[len(PREFIX):];parent,_,field=leaf.rpartition('.')
            module=self.graph.model.get_submodule(parent) if parent else self.graph.model
            if parent in ORDER:module=module.source
            setattr(module,field,nn.Parameter(self.source.tensor(name),requires_grad=False))
        self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
        self.projections={name:FullLinear(name,self.graph.model.get_submodule(name).source,self.executor,self.directory) for name in ORDER}

    def prepare(self):
        installed=versions()
        if json.loads(ext.default_params())!=POLICY['parameters']:raise ValueError('Pinned native parameters required')
        write_component(self.directory,CONFIG,[self.projections[name].definition for name in ORDER],dict(programId=PROGRAM_ID),installed,source=WEIGHTS)
        return load_plan(self.directory)

    def evaluate(self,value,state,context,mode,bridge=None,secret=None):
        if mode not in ['float-reference','clear-reference','execute']:raise ValueError('Explicit QSA execution mode required')
        total=state.tokens+value.shape[1];past=state.tokens
        positions=text_rotary(self.reference_path,self.config,total)
        visible=torch.arange(total)[None,:]<=torch.arange(past,total)[:,None]
        mask=torch.full((1,1,total-past,total),torch.finfo(torch.float32).min).masked_fill(visible[None,None],0)
        if mode=='float-reference':return self.graph(value,state,context,position_embeddings=positions,attention_mask=mask,execution='public-reference')
        def project(operation,value):
            projection=self.projections[operation['id']]
            if (operation['inputWidth'],operation['outputWidth'])!=(projection.definition['inputWidth'],projection.definition['outputWidth']):raise ValueError('Actual QSA projection geometry changed')
            return projection(value)
        for projection in self.projections.values():
            projection.mode=mode;projection.bridge=bridge;projection.secret=secret;projection.parameters=self.executor.glwe_crypto_params
        try:return self.graph(value,state,context,position_embeddings=positions,attention_mask=mask,execution='verified-projections',verified_client_projection=project)
        finally:
            for projection in self.projections.values():projection.secret=projection.bridge=projection.parameters=None


def run(source_dir,cache_dir,reference_path):
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit QSA CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);began=time.perf_counter()
    model=ActualFlashQSA(source_dir,cache_dir,reference_path);plan=model.prepare();width=model.graph.hidden_width
    values=torch.sin(torch.arange(7*width,dtype=torch.float32)*.013).reshape(1,7,width)*.1
    inputs=[values[:,:4],values[:,4:5],values[:,5:6],values[:,6:7]];references={}
    for mode in ['float-reference','clear-reference']:
        context=ClientContext('public-fixture',plan.identity,plan.definition()['modelId'],mode,mode);state=AttentionRequestState(context,layer_index=3)
        try:references[mode]=[model.evaluate(value,state,context,mode) for value in inputs]
        finally:state.close()
    secret,key=ext.cpu_create_private_key(model.executor.glwe_crypto_params);bridge=None;state=None
    try:
        bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'])
        context=ClientContext('public-fixture',plan.identity,plan.definition()['modelId'],bridge.key_id,bridge.session)
        state=AttentionRequestState(context,layer_index=3);errors=[];step_ms=[]
        def execute(index):
            started=time.perf_counter();actual=model.evaluate(inputs[index],state,context,'execute',bridge,secret)
            step_ms.append((time.perf_counter()-started)*1000)
            return actual
        def validate(index,actual):
            if not torch.allclose(actual,references['clear-reference'][index],rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):raise ExecutionRejected('component-reference-mismatch')
            errors.append(float((actual-references['clear-reference'][index]).abs().max()))
        complete_verified_route(bridge,len(inputs),execute,validate)
        return dict(schema='doppler.actual-flash-qsa-execution/v1',fixtureId='flash-first-qsa-four-public-inputs',
                    scope='Actual layer-3 QSA; four cached public-input steps; encrypted linear kernels with client indexing, attention, rotary, nonlinear operations and full replay',
                    productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                    sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],programId=PROGRAM_ID,modelId=plan.definition()['modelId'],planId=plan.identity,protocol=plan.protocol_id,
                    validationBoundary=VALIDATION_BOUNDARY,componentSteps=CONFIG['steps'],linearModules=len(ORDER),encryptedTiles=bridge.sequence-1,quantizedReferenceCheckPassed=True,
                    maxEncryptedVsQuantizedErrors=errors,maxQuantizedVsFloatingErrors=[float((a-b).abs().max()) for a,b in zip(references['clear-reference'],references['float-reference'])],
                    stepMs=step_ms,elapsedMs=(time.perf_counter()-began)*1000,evaluationKeyBytes=bridge.key_bytes,sentBytes=bridge.sent,receivedBytes=bridge.received,
                    clientEncryptedReplayMs=bridge.replay_seconds*1000,evaluatorRoundTripMs=bridge.remote_seconds*1000,
                    clientLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),implementationVersions=versions())
    finally:
        try:
            if state is not None:state.close()
            if bridge is not None:bridge.close()
        finally:secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True);parser.add_argument('--reference-source',required=True)
    args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir,args.reference_source)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-qsa-execution/v1',productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True)
        raise
    print(json.dumps(report,indent=2))
