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
from full_plan import canonical
from flash_component_plan import LIMITS
from flash_moe_plan import CONFIG,PROGRAM,SOURCE_ID,WEIGHTS,lowered_operations,load_plan,versions
from flash_weights import FlashSource
from full_status import ExecutionRejected,failure_event
from protocol import POLICY,digest

PREFIX='model.language_model.layers.0.mlp.'


class StreamedMoE:
    """Only one public expert's FP32 material is resident at once; private state is attempt-local."""
    def __init__(self,source_dir,cache_dir):
        self.source=FlashSource(source_dir)
        self.directory=Path(cache_dir);self.directory.mkdir(parents=True,exist_ok=True)
        self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)

    def projection(self,name):
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

    def prepare(self):
        installed=versions()
        if json.loads(ext.default_params())!=POLICY['parameters']:raise ValueError('Pinned native parameters required')
        definitions=[]
        for row in lowered_operations():
            projection=self.projection(row['name'])
            if any(projection.definition[key]!=value for key,value in row.items()):raise ValueError('Flash projection lowering changed')
            definitions.append(projection.definition)
            del projection
        semantics=dict(programId=PROGRAM.identity)
        plan=dict(schema=CONFIG['planSchema'],protocol=CONFIG['protocol'],sourceId=SOURCE_ID,source=WEIGHTS,
                  modelId=digest(canonical(dict(sourceId=SOURCE_ID,**semantics,projections=definitions))),
                  productionBackendAvailable=False,implementationVersions=installed,
                  parameters={key:POLICY[key] for key in ['dependencies','parameters','evaluationBackend']},
                  **semantics,**{key:CONFIG[key] for key in LIMITS},projections=definitions)
        (self.directory/'plan.json').write_bytes(canonical(plan))
        return load_plan(self.directory)

    def evaluate(self,hidden,mode,plan,bridge=None,secret=None):
        expected=iter(plan.definition()['projections']);pending=None;position=0
        def project(operation,values):
            nonlocal pending,position
            name=operation['id']
            if name.startswith('expert/') and name.endswith('/up'):
                if pending is None or pending[0]!=name:raise ValueError('Flash combined projection order changed')
                result=pending[1];pending=None
                return result
            if pending is not None:raise ValueError('Incomplete Flash combined projection')
            combined=name.startswith('expert/') and name.endswith('/gate')
            lowered=name[:-4]+'gate_up' if combined else name
            definition=next(expected);projection=self.projection(lowered)
            if projection.definition!=definition:raise ValueError('Public Flash projection changed after admission')
            projection.mode=mode;projection.bridge=bridge;projection.secret=secret
            projection.parameters=self.executor.glwe_crypto_params
            try:
                result=projection(torch.tensor(values,dtype=torch.float32).reshape(1,-1)).reshape(-1).tolist()
            finally:projection.secret=projection.bridge=projection.parameters=None
            position+=1
            if combined:
                width=operation['outputWidth']
                pending=(name[:-4]+'up',result[width:]);result=result[:width]
            return result
        with torch.no_grad():result=PROGRAM.attempt().run(hidden,project)
        if pending is not None or position!=len(plan.definition()['projections']):raise ValueError('Incomplete Flash lowered route')
        return torch.tensor(result,dtype=torch.float32)


def run(source_dir,cache_dir,execution):
    if execution not in ['public-reference','encrypted']:raise ValueError('Explicit Flash execution mode required')
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit component CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);began=time.perf_counter()
    model=StreamedMoE(source_dir,cache_dir);plan=model.prepare()
    # An explicitly public fixture; no private input channel is supported by this experiment.
    hidden=torch.sin(torch.arange(PROGRAM.experts.hidden_width,dtype=torch.float32)*.013).tolist()
    floating=model.evaluate(hidden,'float-reference',plan)
    quantized=model.evaluate(hidden,'clear-reference',plan)
    report=dict(schema='doppler.actual-flash-dense-moe-execution/v1',fixtureId='flash-dense-moe-sine-2560',
                productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                sourceId=SOURCE_ID,sourceRevision=WEIGHTS['revision'],programId=PROGRAM.identity,planId=plan.identity,
                modelId=plan.definition()['modelId'],protocol=plan.protocol_id,expertCount=PROGRAM.experts.expert_count,
                topK=PROGRAM.experts.top_k,logicalProjections=len(PROGRAM.contract()['operations']),
                linearModules=len(plan.definition()['projections']),plannedEncryptedTiles=len(plan.schedule()),
                allExpertsVisited=True,encryptedExecutionCompleted=False,encryptedTiles=0,
                maxQuantizedVsFloatingError=float((quantized-floating).abs().max()),execution=execution)
    if execution=='public-reference':return dict(report,elapsedMs=(time.perf_counter()-began)*1000)
    secret,key=ext.cpu_create_private_key(model.executor.glwe_crypto_params);bridge=None
    try:
        bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'],
                          observe=lambda event,value:print(json.dumps(dict(event=event,**value)),file=sys.stderr,flush=True))
        started=time.perf_counter();actual=model.evaluate(hidden,'execute',plan,bridge,secret)
        if not torch.allclose(actual,quantized,rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):
            raise ExecutionRejected('component-reference-mismatch')
        bridge.finish()
        return dict(report,encryptedExecutionCompleted=True,encryptedTiles=bridge.sequence-1,
                    maxEncryptedVsQuantizedError=float((actual-quantized).abs().max()),
                    quantizedReferenceCheckPassed=True,executeMs=(time.perf_counter()-started)*1000,
                    elapsedMs=(time.perf_counter()-began)*1000,evaluationKeyBytes=bridge.key_bytes,
                    sentBytes=bridge.sent,receivedBytes=bridge.received,clientEncryptedReplayMs=bridge.replay_seconds*1000,
                    evaluatorRoundTripMs=bridge.remote_seconds*1000)
    finally:
        if bridge is not None:bridge.close()
        secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--execution',required=True,choices=['public-reference','encrypted']);args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir,args.execution)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-dense-moe-execution/v1',productionBackendAvailable=False,
                              securityQualified=False,fullFlashModelExecuted=False,
                              status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True)
        raise
    print(json.dumps(report,indent=2))
