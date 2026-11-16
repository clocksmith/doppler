"""Wired actual Flash PLE candidate; complete local table is mandatory before key disclosure.

No table download, remote row lookup or fallback exists. This path has not executed
with the actual 104 GB table. It remains outside production admission.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import sys
import torch
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
import concrete_ml_extensions as ext
from full_model import FullLinear
from full_bridge import FullBridge
from full_status import ExecutionRejected,failure_event
from flash_weights import FlashSource
from flash_component_plan import write_component
from flash_ple import ClientPLE
from flash_ple_state import PLERequestState
from flash_request_state import ClientContext
from local_ngram_rows import LocalNgramRows
from flash_ple_plan import CONFIG,TABLE,TABLE_ID,WEIGHTS,SOURCE_ID,PREFIX,ORDER,SEMANTICS,load_plan,versions
from protocol import digest
from full_plan import canonical


class ActualFlashPLE:
    def __init__(self,source_dir,cache_dir,reference_path,*,local_storage_limit_bytes):
        versions() # Unsupported native variants fail before costly public preparation.
        self.rows=LocalNgramRows.flash(source_dir,local_storage_limit_bytes=local_storage_limit_bytes)
        try:
            if digest(canonical(self.rows.spec))!=TABLE_ID:raise ValueError('Installed complete local table identity changed')
            self.source=FlashSource(source_dir,scope='ple');self.directory=Path(cache_dir);self.directory.mkdir(parents=True,exist_ok=True)
            config=dict(self.source.config['text_config'])
            for key,value in TABLE['resolvedNgramDefaults'].items():
                if key in config and config[key]!=value:raise ValueError('Pinned PLE configuration default changed')
                config[key]=value
            self.config=SimpleNamespace(**config)
            parameters={name[len(PREFIX):]:self.source.tensor(name) for name in WEIGHTS['tensors']}
            self.graph=ClientPLE.from_local_rows(reference_path,self.config,self.rows,parameters,
                context_limit=8,max_lookup_bytes=CONFIG['maxLookupBytes'])
            self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
            self.projections={name:FullLinear(name,getattr(self.graph.model,name).source,self.executor,self.directory) for name in ORDER}
        except BaseException:
            self.close();raise

    def prepare(self):
        write_component(self.directory,CONFIG,[self.projections[name].definition for name in ORDER],SEMANTICS,versions(),source=WEIGHTS)
        return load_plan(self.directory)

    def evaluate(self,value,tokens,state,context,mode,bridge=None,secret=None):
        if mode=='float-reference':return self.graph(value,tokens,state,context,execution='public-reference')
        if mode not in ['clear-reference','execute']:raise ValueError('Explicit PLE execution mode required')
        def project(operation,value):
            projection=self.projections[operation['id']]
            if (operation['inputWidth'],operation['outputWidth'])!=(projection.definition['inputWidth'],projection.definition['outputWidth']):raise ValueError('PLE projection geometry changed')
            return projection(value)
        for projection in self.projections.values():
            projection.mode=mode;projection.bridge=bridge;projection.secret=secret;projection.parameters=self.executor.glwe_crypto_params
        try:return self.graph(value,tokens,state,context,execution='verified-projections',verified_client_projection=project)
        finally:
            for projection in self.projections.values():projection.secret=projection.bridge=projection.parameters=None

    def close(self):self.rows.close()


def run(source_dir,cache_dir,reference_path,*,local_storage_limit_bytes,prepare_only=False):
    if os.environ.get('RAYON_NUM_THREADS')!=str(CONFIG['cpuThreads']):raise ValueError('Explicit PLE CPU budget required')
    torch.set_num_threads(CONFIG['cpuThreads']);began=time.perf_counter()
    model=ActualFlashPLE(source_dir,cache_dir,reference_path,local_storage_limit_bytes=local_storage_limit_bytes)
    bridge=state=None;secret=key=None
    try:
        plan=model.prepare()
        if prepare_only:return dict(planId=plan.identity,localTableId=TABLE_ID,productionBackendAvailable=False,encryptedTiles=0)
        width=model.graph.hidden_width;values=torch.sin(torch.arange(7*width,dtype=torch.float32)*.013).reshape(1,7,width)*.1
        eos=model.config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
        tokens=torch.tensor([[1,2,3,eos,4,5,6]],dtype=torch.int64)
        inputs=[(values[:,:4],tokens[:,:4])]+[(values[:,i:i+1],tokens[:,i:i+1]) for i in range(4,7)]
        references={}
        for mode in ['float-reference','clear-reference']:
            context=ClientContext('public-fixture',plan.identity,plan.definition()['modelId'],mode,mode);reference_state=PLERequestState(context)
            try:references[mode]=[model.evaluate(value,ids,reference_state,context,mode) for value,ids in inputs]
            finally:reference_state.close()
        secret,key=ext.cpu_create_private_key(model.executor.glwe_crypto_params);bridge=FullBridge(cache_dir,plan,key,backend_profile=CONFIG['profile'])
        context=ClientContext('public-fixture',plan.identity,plan.definition()['modelId'],bridge.key_id,bridge.session);state=PLERequestState(context);errors=[]
        for index,(value,ids) in enumerate(inputs):
            actual=model.evaluate(value,ids,state,context,'execute',bridge,secret)
            if not torch.allclose(actual,references['clear-reference'][index],rtol=CONFIG['maxRelativeError'],atol=CONFIG['maxAbsoluteError']):raise ExecutionRejected('component-reference-mismatch')
            errors.append(float((actual-references['clear-reference'][index]).abs().max()))
        bridge.finish()
        return dict(schema='doppler.actual-flash-ple-execution/v1',fixtureId='flash-first-ple-four-public-inputs',
                    productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,
                    sourceId=SOURCE_ID,localTableId=TABLE_ID,planId=plan.identity,modelId=plan.definition()['modelId'],protocol=plan.protocol_id,
                    encryptedTiles=bridge.sequence-1,componentSteps=4,linearModules=2,quantizedReferenceCheckPassed=True,maxEncryptedVsQuantizedErrors=errors,
                    maxQuantizedVsFloatingErrors=[float((a-b).abs().max()) for a,b in zip(references['clear-reference'],references['float-reference'])],
                    elapsedMs=(time.perf_counter()-began)*1000,evaluationKeyBytes=bridge.key_bytes,sentBytes=bridge.sent,receivedBytes=bridge.received,
                    clientEncryptedReplayMs=bridge.replay_seconds*1000,evaluatorRoundTripMs=bridge.remote_seconds*1000)
    finally:
        try:
            if state is not None:state.close()
            if bridge is not None:bridge.close()
        finally:
            secret=key=None;model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--reference-source',required=True);parser.add_argument('--local-storage-limit-bytes',type=int,required=True);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.source_dir,args.cache_dir,args.reference_source,local_storage_limit_bytes=args.local_storage_limit_bytes,prepare_only=args.prepare_only)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.actual-flash-ple-execution/v1',productionBackendAvailable=False,securityQualified=False,fullFlashModelExecuted=False,status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True);raise
    print(json.dumps(report,indent=2))
