"""Stream all actual Flash experts through one fixed encrypted route; public fixture only."""
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
from verified_execution import complete_verified_route,VALIDATION_BOUNDARY
from flash_component_plan import write_component
from flash_moe_plan import CONFIG,PROGRAM,SOURCE_ID,WEIGHTS,lowered_operations
from flash_weights import FlashSource
from full_status import ExecutionRejected,failure_event
from protocol import POLICY,require_active
from plan_backends import moe_backend,MOE_PROFILES

PREFIX='model.language_model.layers.0.mlp.'


class StreamedMoE:
    """Only one public expert's FP32 material is resident at once; private state is attempt-local."""
    def __init__(self,source_dir,cache_dir,backend_profile=CONFIG['profile'],*,cancelled=None):
        require_active(cancelled)
        self.config,self.plan_loader,self.version_check=moe_backend(backend_profile)
        self.implementation=self.version_check()
        require_active(cancelled)
        self.source=FlashSource(source_dir,cancelled=cancelled)
        require_active(cancelled)
        self.directory=Path(cache_dir);self.directory.mkdir(parents=True,exist_ok=True)
        self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True);self.prepared={}

    def projection(self,name,mode='float-reference'):
        if mode not in ['float-reference','clear-reference','execute']:raise ValueError('Explicit Flash projection mode required')
        if mode!='float-reference':
            if name not in self.prepared:raise ValueError('Complete client projection preparation required')
            return self.prepared[name].prepared_copy()
        if name.startswith('expert/'):
            _,index,leaf=name.split('/')
            weight=self.source.tensor(PREFIX+'experts.'+leaf+'_proj',expert=int(index))
        else:
            leaf=name.split('/')[-1]
            tensor={'router/logits':'gate.weight','shared/mix':'shared_expert_gate.weight'}.get(name)
            weight=self.source.tensor(PREFIX+(tensor if tensor is not None else 'shared_expert.'+leaf+'_proj.weight'))
        with torch.device('meta'):source=nn.Linear(weight.shape[1],weight.shape[0],bias=False)
        source.weight=nn.Parameter(weight,requires_grad=False)
        return FullLinear(name,source,self.executor,self.directory)

    def prepare(self,*,cancelled=None):
        require_active(cancelled)
        installed=self.implementation
        if json.loads(ext.default_params())!=POLICY['parameters']:raise ValueError('Pinned native parameters required')
        definitions=self.prepare_projections(lowered_operations(),cancelled=cancelled)
        require_active(cancelled)
        write_component(self.directory,self.config,definitions,dict(programId=PROGRAM.identity),installed)
        plan=self.plan_loader(self.directory)
        require_active(cancelled)
        return plan

    def prepare_projections(self,operations,*,cancelled=None):
        require_active(cancelled)
        self.prepared={};prepared={};definitions=[];operations=tuple(operations)
        names=[row['name'] for row in operations]
        if len(names)!=len(set(names)):raise ValueError('Distinct prepared Flash projection names required')
        for row in operations:
            require_active(cancelled)
            projection=self.projection(row['name'])
            require_active(cancelled)
            if any(projection.definition[key]!=value for key,value in row.items()):raise ValueError('Flash projection lowering changed')
            definitions.append(dict(projection.definition))
            projection.release_float_reference();prepared[projection.name]=projection
        require_active(cancelled)
        self.prepared=prepared
        return definitions

    def evaluate(self,hidden,mode,plan,bridge=None,secret=None,projection_definitions=None,*,cancelled=None):
        if (mode not in ['float-reference','clear-reference','execute']
                or (mode=='execute' and (bridge is None or secret is None))
                or (mode!='execute' and (bridge is not None or secret is not None))
                or (bridge is not None and cancelled is not None and bridge.cancelled is not cancelled)):
            raise ValueError('Explicit matching Flash execution and cancellation context required')
        def check_active():
            if bridge is not None:bridge.check_active()
            require_active(cancelled)
        check_active()
        definitions=plan.definition()['projections'] if projection_definitions is None else projection_definitions
        expected=iter(definitions);pending=None;position=0
        def project(operation,values):
            nonlocal pending,position
            check_active()
            name=operation['id']
            if name.startswith('expert/') and name.endswith('/up'):
                if pending is None or pending[0]!=name:raise ValueError('Flash combined projection order changed')
                result=pending[1];pending=None
                return result
            if pending is not None:raise ValueError('Incomplete Flash combined projection')
            combined=name.startswith('expert/') and name.endswith('/gate')
            lowered=name[:-4]+'gate_up' if combined else name
            definition=next(expected);projection=self.projection(lowered,mode)
            check_active()
            if projection.definition!=definition:raise ValueError('Public Flash projection changed after admission')
            projection.mode=mode;projection.bridge=bridge;projection.secret=secret
            projection.parameters=self.executor.glwe_crypto_params
            try:
                result=projection(torch.tensor(values,dtype=torch.float32).reshape(1,-1)).reshape(-1).tolist()
                check_active()
            finally:projection.secret=projection.bridge=projection.parameters=None
            position+=1
            if combined:
                width=operation['outputWidth']
                pending=(name[:-4]+'up',result[width:]);result=result[:width]
            return result
        with torch.no_grad():result=PROGRAM.attempt().run(hidden,project)
        check_active()
        if pending is not None or position!=len(definitions):raise ValueError('Incomplete Flash lowered route')
        return torch.tensor(result,dtype=torch.float32)


def run(source_dir,cache_dir,execution,backend_profile=CONFIG['profile'],*,cancelled=None):
    require_active(cancelled)
    if execution not in ['public-reference','encrypted']:raise ValueError('Explicit Flash execution mode required')
    config,_,_=moe_backend(backend_profile)
    if os.environ.get('RAYON_NUM_THREADS')!=str(config['cpuThreads']):raise ValueError('Explicit component CPU budget required')
    torch.set_num_threads(config['cpuThreads']);began=time.perf_counter()
    model=StreamedMoE(source_dir,cache_dir,backend_profile,cancelled=cancelled);plan=model.prepare(cancelled=cancelled)
    # An explicitly public fixture; no private input channel is supported by this experiment.
    hidden=torch.sin(torch.arange(PROGRAM.experts.hidden_width,dtype=torch.float32)*.013).tolist()
    floating=model.evaluate(hidden,'float-reference',plan,cancelled=cancelled)
    quantized=model.evaluate(hidden,'clear-reference',plan,cancelled=cancelled)
    report=dict(schema='doppler.actual-flash-dense-moe-execution/v1',fixtureId='flash-dense-moe-sine-2560',
                productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],programId=PROGRAM.identity,planId=plan.identity,
                modelId=plan.definition()['modelId'],protocol=plan.protocol_id,backendProfile=backend_profile,implementationVersions=model.implementation,expertCount=PROGRAM.experts.expert_count,
                topK=PROGRAM.experts.top_k,logicalProjections=len(PROGRAM.contract()['operations']),
                linearModules=len(plan.definition()['projections']),plannedEncryptedTiles=len(plan.schedule()),
                allExpertsVisited=True,encryptedExecutionCompleted=False,encryptedTiles=0,
                maxQuantizedVsFloatingError=float((quantized-floating).abs().max()),execution=execution)
    require_active(cancelled)
    if execution=='public-reference':return dict(report,elapsedMs=(time.perf_counter()-began)*1000)
    secret,key=ext.cpu_create_private_key(model.executor.glwe_crypto_params);bridge=None
    try:
        require_active(cancelled)
        bridge=FullBridge(cache_dir,plan,key,backend_profile=backend_profile,cancelled=cancelled,
                          observe=lambda event,value:print(json.dumps(dict(event=event,**value)),file=sys.stderr,flush=True))
        def validate(step,actual):
            require_active(cancelled)
            if not torch.allclose(actual,quantized,rtol=config['maxRelativeError'],atol=config['maxAbsoluteError']):
                raise ExecutionRejected('component-reference-mismatch')
        started=time.perf_counter()
        actual=complete_verified_route(bridge,1,lambda step:model.evaluate(hidden,'execute',plan,bridge,secret,cancelled=cancelled),validate)[0]
        return dict(report,encryptedExecutionCompleted=True,validationBoundary=VALIDATION_BOUNDARY,encryptedTiles=bridge.sequence-1,
                    maxEncryptedVsQuantizedError=float((actual-quantized).abs().max()),
                    quantizedReferenceCheckPassed=True,executeMs=(time.perf_counter()-started)*1000,
                    elapsedMs=(time.perf_counter()-began)*1000,evaluationKeyBytes=bridge.key_bytes,
                    sentBytes=bridge.sent,receivedBytes=bridge.received,clientEncryptedReplayMs=bridge.replay_seconds*1000,
                    evaluatorRoundTripMs=bridge.remote_seconds*1000)
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--execution',required=True,choices=['public-reference','encrypted'])
    parser.add_argument('--backend-profile',choices=MOE_PROFILES,default=CONFIG['profile']);args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir,args.execution,args.backend_profile)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-dense-moe-execution/v1',productionBackendAvailable=False,
                              securityQualified=False,fullFlashModelExecuted=False,
                              status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True)
        raise
    print(json.dumps(report,indent=2))
