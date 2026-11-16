"""Actual Flash gated-residual encrypted feedback fixture; not full Flash autoregression."""
import argparse
import json
import os
from pathlib import Path
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
from flash_hyper_ops import TorchHyperConnection
from flash_hyper_program import require_source_config
from flash_hyper_plan import CONFIG,WEIGHTS,SOURCE_ID,PROGRAM,load_plan,versions
from flash_weights import FlashSource
from protocol import POLICY

PREFIX='model.language_model.layers.0.mlp_hyper_connection.'
PROJECTIONS={'hyper/down':'input_mix_weight_down.weight','hyper/up':'input_mix_weight_up.weight',
             'hyper/inject':'block_inject_weight.weight'}


class FlashHyperModel:
    def __init__(self,source_dir,cache_dir):
        self.source=FlashSource(source_dir,scope='mlp-hyper')
        require_source_config(PROGRAM,self.source.config['text_config'])
        self.directory=Path(cache_dir);self.directory.mkdir(parents=True,exist_ok=True)
        self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
        self.projections=[]
        for operation in PROGRAM.contract()['operations']:
            weight=self.source.tensor(PREFIX+PROJECTIONS[operation['id']])
            with torch.device('meta'):linear=nn.Linear(weight.shape[1],weight.shape[0],bias=False)
            linear.weight=nn.Parameter(weight,requires_grad=False)
            self.projections.append(FullLinear(operation['id'],linear,self.executor,self.directory))
        by_name={projection.name:projection for projection in self.projections}
        self.graph=TorchHyperConnection(PROGRAM,self.source.tensor(PREFIX+'hc_norm.weight'),
                                        lambda operation,value:by_name[operation['id']](value))

    def prepare(self):
        installed=versions()
        if json.loads(ext.default_params())!=POLICY['parameters']:raise ValueError('Pinned native parameters required')
        definitions=[projection.definition for projection in self.projections]
        write_component(self.directory,CONFIG,definitions,dict(programId=PROGRAM.identity),installed,source=WEIGHTS)
        return load_plan(self.directory)

    def evaluate(self,value,mode,bridge=None,secret=None,block=None):
        for projection in self.projections:
            projection.mode=mode;projection.bridge=bridge;projection.secret=secret
            projection.parameters=self.executor.glwe_crypto_params
        try:
            with torch.no_grad():
                mixed,residual,injection=self.graph(value)
                output=mixed if block is None else block(mixed)
                return self.graph.combine(residual,injection,output),mixed,injection
        finally:
            for projection in self.projections:projection.secret=projection.bridge=projection.parameters=None


def run(source_dir,cache_dir):
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit hyperconnection CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);began=time.perf_counter()
    model=FlashHyperModel(source_dir,cache_dir);plan=model.prepare()
    fixture=torch.sin(torch.arange(PROGRAM.streams*PROGRAM.hidden_width,dtype=torch.float32)*.013).reshape(1,-1)*.1
    states={mode:fixture.clone() for mode in ['float-reference','clear-reference','execute']}
    secret,key=ext.cpu_create_private_key(model.executor.glwe_crypto_params);bridge=None
    try:
        bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'],
                          observe=lambda event,value:print(json.dumps(dict(event=event,**value)),file=sys.stderr,flush=True))
        errors=[];quantized_errors=[];step_ms=[]
        for _step in range(CONFIG['steps']):
            results={}
            for mode in states:
                started=time.perf_counter();results[mode]=model.evaluate(states[mode],mode,bridge,secret)
                states[mode]=results[mode][0]
                if mode=='execute':step_ms.append((time.perf_counter()-started)*1000)
            for actual,expected in zip(results['execute'],results['clear-reference']):
                if not torch.allclose(actual,expected,rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):
                    raise ExecutionRejected('component-reference-mismatch')
            errors.append(max(float((actual-expected).abs().max()) for actual,expected in zip(results['execute'],results['clear-reference'])))
            quantized_errors.append(float((states['clear-reference']-states['float-reference']).abs().max()))
        bridge.finish()
        return dict(schema='doppler.actual-flash-hyper-execution/v1',fixtureId='flash-mlp-hyper-identity-feedback',
                    scope='Four public gated-residual identity-block updates; no attention, MoE or Flash token generation',
                    productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                    sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],programId=PROGRAM.identity,
                    modelId=plan.definition()['modelId'],planId=plan.identity,protocol=plan.protocol_id,
                    componentSteps=CONFIG['steps'],linearModules=len(model.projections),encryptedTiles=bridge.sequence-1,
                    quantizedReferenceCheckPassed=True,maxEncryptedVsQuantizedErrors=errors,
                    maxQuantizedVsFloatingErrors=quantized_errors,stepMs=step_ms,elapsedMs=(time.perf_counter()-began)*1000,
                    evaluationKeyBytes=bridge.key_bytes,sentBytes=bridge.sent,receivedBytes=bridge.received,
                    clientEncryptedReplayMs=bridge.replay_seconds*1000,evaluatorRoundTripMs=bridge.remote_seconds*1000)
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-hyper-execution/v1',productionBackendAvailable=False,
                              securityQualified=False,fullFlashModelExecuted=False,
                              status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True)
        raise
    print(json.dumps(report,indent=2))
