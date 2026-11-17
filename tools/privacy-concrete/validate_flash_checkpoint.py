"""Open the complete source-bound Flash text graph over an actual local checkpoint
and run clear cached generation through it.

Every decoder stage, the output stage and both local row stores are admitted by
the same loaders an encrypted attempt uses: whole-file SHA-256, pinned geometry
and complete tensor accounting. Execution here is clear and client-local. There
are no keys, ciphertexts or evaluator, and no production or security claim.

`--mode quantized` adds a twelve-bit integer reference that quantizes each
projection as it is borrowed from source and hashes the coefficient bytes, so
the ordered coefficient inventory is bound without writing a coefficient cache.
"""
import argparse
import fcntl
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import resource
import struct
import sys
import time
from types import SimpleNamespace

SCHEMA='doppler.actual-flash-source-bound-validation/v1'


def log(event,**fields):print(json.dumps(dict(event=event,at=round(time.time(),3),**fields)),file=sys.stderr,flush=True)


def publish(path,value):
    temporary=Path(str(path)+'.partial');temporary.write_text(json.dumps(value,indent=1)+'\n');temporary.replace(path)


class Turn:
    """Cooperative host lock, held only while this process does heavy CPU or disk work."""
    def __init__(self,path,pause=None):
        self.handle=open(path,'a') if path else None;self.depth=0;self.waited=0.0
        self.pause=Path(pause) if pause else None

    def acquire(self):
        if self.depth==0:
            began=time.perf_counter()
            # Another job on this host can create the pause file to make this
            # process stop before its next unit, then take the lock without a race.
            while self.pause is not None and self.pause.exists():time.sleep(5)
            if self.handle is not None:fcntl.flock(self.handle,fcntl.LOCK_EX)
            self.waited+=time.perf_counter()-began
        self.depth+=1

    def release(self):
        self.depth-=1
        if self.handle is not None and self.depth==0:fcntl.flock(self.handle,fcntl.LOCK_UN)


def metadata(source):
    """Sizes, small-file hashes and safetensors headers only; no weight bytes are read."""
    from flash_source_scopes import TEXT
    source=Path(source);table=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())
    for name,record in TEXT['files'].items():
        data=(source/name).read_bytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Pinned checkpoint file differs: '+name)
    headers={};total=0
    for name,record in TEXT['shards'].items():
        path=source/name;size=path.stat().st_size;total+=size
        if size!=record['bytes']:raise ValueError('Checkpoint shard size differs: '+name)
        with path.open('rb') as handle:
            length=struct.unpack('<Q',handle.read(8))[0]
            if not 0<length<=16*1024*1024:raise ValueError('Bounded safetensors header required')
            headers[name]=json.loads(handle.read(length))
    for name,expected in TEXT['tensors'].items():
        actual=headers[expected['shard']].get(name)
        if (not isinstance(actual,dict) or actual.get('dtype')!=expected['dtype'] or actual.get('shape')!=expected['shape']
                or actual.get('data_offsets')!=expected['dataOffsets']):raise ValueError('Checkpoint tensor differs from the installed inventory: '+name)
    for name,record in table['shards'].items():
        if (source/name).stat().st_size!=record['bytes']:raise ValueError('Local table shard size differs: '+name)
    return dict(shards=len(TEXT['shards']),tensors=len(TEXT['tensors']),shardBytes=total,localTableShards=len(table['shards']),
                declaredShardBytes=TEXT['wholeShardBytes'],weightBytesRead=0)


class StreamedQuantized:
    """Clear integer projection with the encrypted route's quantizer; nothing is stored."""
    def __init__(self,graph,bits):
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        from weight_encoding import encoding_for_bits
        self.graph=graph;self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True);self.encoding=encoding_for_bits(bits)
        self.context=SimpleNamespace(input_quantizers=[SimpleNamespace(quant_options=SimpleNamespace(n_bits=bits))])
        self.identities={};self.coefficients=0

    def __call__(self,operation,value):
        import numpy as np
        import torch
        from integer_reference import integer_dot
        name=operation['id'];source=self.graph.projections[name].source
        with source.materialized_weight() as weight:
            quantized,scale,zero_point,sum_weights=self.executor._per_channel_weight_quantization(weight.detach().numpy().T,self.context,torch.device('cpu'))
        data=np.ascontiguousarray(quantized.numpy(),dtype=self.encoding['dtype'])
        if data.shape!=(operation['inputWidth'],operation['outputWidth']) or not np.array_equal(data,quantized.numpy()):raise ValueError('Quantized public coefficient truncation')
        identity=hashlib.sha256(data.tobytes()).hexdigest()
        if self.identities.setdefault(name,identity)!=identity:raise ValueError('Quantized coefficient identity changed between steps')
        if len(self.identities)>self.coefficients:self.coefficients=len(self.identities)
        q_input,input_scale,input_zero,shape=self.executor._dynamic_input_quantization(value,self.context,False)
        raw=integer_dot(q_input,data)
        output=self.executor._apply_correction_and_dequantize(raw,q_input,input_zero,zero_point,sum_weights,operation['inputWidth'],input_scale,scale)
        return output.view(*shape[:-1],operation['outputWidth'])

    def inventory(self,order):
        if set(self.identities)!=set(order):raise ValueError('Incomplete quantized coefficient inventory')
        ordered=[dict(name=name,weightId=self.identities[name]) for name in order]
        return ordered,hashlib.sha256(json.dumps(ordered,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run(args):
    import numpy as np
    import torch
    import flash_tokenizer
    from flash_graph_source import bind_graph
    from flash_reference import provision_reference
    from flash_request_state import ClientContext
    from flash_source_scopes import TEXT
    from flash_stage_source import FlashStageSource
    from full_plan import canonical
    from local_ngram_rows import LocalNgramRows
    from local_token_rows import LocalTokenRows
    from protocol import digest
    work=Path(args.work);work.mkdir(parents=True,exist_ok=True);report_path=work/'report.json'
    torch.set_num_threads(args.threads);turn=Turn(args.lock,args.pause_file);started=time.perf_counter()
    report=dict(schema=SCHEMA,repository=TEXT['repository'],revision=TEXT['revision'],sourceId=digest(canonical(TEXT)),completed=False,
                encryptedExecution=False,productionBackendAvailable=False,securityQualified=False,cpuThreads=args.threads,steps=args.steps)
    # Wait for the host turn before touching the checkpoint at all, headers included.
    turn.acquire()
    try:report['metadata']=metadata(args.source)
    finally:turn.release()
    publish(report_path,report);log('metadata-verified',**report['metadata'])
    provision_reference(args.reference);reference=Path(args.reference)/'modeling_qwen4_exp.py'
    with ExitStack() as stack:
        decoders=[];began=time.perf_counter();verified=0
        for index in range(TEXT['layerCount']):
            turn.acquire()
            try:source=FlashStageSource(args.source,'decoder',index)
            finally:turn.release()
            stack.callback(source.close);decoders.append(source);verified+=source.spec['wholeShardBytes']
            log('decoder-stage-verified',layer=index,shardBytes=source.spec['wholeShardBytes'],elapsedS=round(time.perf_counter()-began,1))
        turn.acquire()
        try:
            output=FlashStageSource(args.source,'output');stack.callback(output.close);verified+=output.spec['wholeShardBytes']
            token_rows=LocalTokenRows.flash(args.source,local_storage_limit_bytes=4*1024**3);stack.callback(token_rows.close)
            ple_rows=LocalNgramRows.flash(args.source,local_storage_limit_bytes=128*1024**3);stack.callback(ple_rows.close)
            verified+=token_rows.spec['wholeShardBytes']+ple_rows.spec['wholeShardBytes']
            graph=bind_graph(decoders,output,token_rows,ple_rows,reference,context_limit=args.context_limit,
                             max_lookup_rows=args.context_limit,max_ple_lookup_bytes=64*1024**2)
        finally:turn.release()
        report.update(admission=dict(decoderStages=len(decoders),outputStage=True,localTokenRows=True,localTableRows=True,
            sha256VerifiedBytesIncludingRepeats=verified,seconds=round(time.perf_counter()-began,1)),
            clientGraphId=graph.identity,projections=len(graph.order),layerTypes=list(graph.config.layer_types))
        publish(report_path,report);log('graph-bound',clientGraphId=graph.identity,projections=len(graph.order))
        timings={}
        for index,layer in enumerate(graph.layers):
            def enter(module,arguments,index=index):
                turn.acquire();timings[index]=time.perf_counter()
            def leave(module,arguments,result,index=index):
                turn.release();log('layer',layer=index,seconds=round(time.perf_counter()-timings[index],2))
            layer.register_forward_pre_hook(enter);layer.register_forward_hook(leave)
        fixture=flash_tokenizer.SOURCE['publicFixture']
        prompt=list(flash_tokenizer.encode(args.source,fixture['text'],fixture['maxTokens']))
        report.update(publicFixtureId=fixture['id'],promptTokens=prompt)

        def generate(mode,forced=None,callback=None):
            context=ClientContext('public-validation','clear-reference',graph.identity,'no-key',mode+'-'+os.urandom(8).hex())
            state=graph.new_state(context);rows=[];value=torch.tensor([prompt],dtype=torch.int64)
            try:
                for step in range(args.steps):
                    step_began=time.perf_counter()
                    logits=graph(value,state,context,execution='public-reference' if callback is None else 'verified-projections',
                                 verified_client_projection=callback)[0,-1]
                    token=int(logits.argmax());top=torch.topk(logits,5)
                    np.save(work/f'{mode}-step{step}-logits.npy',logits.numpy())
                    rows.append(dict(step=step,inputRows=int(value.shape[1]),seconds=round(time.perf_counter()-step_began,1),greedyToken=token,
                        top5=[[int(i),float(v)] for v,i in zip(top.values,top.indices)],logitsSha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest()))
                    report[mode]=dict(steps=rows);publish(report_path,report);log('step',mode=mode,**rows[-1])
                    value=torch.tensor([[token if forced is None else forced[step]]],dtype=torch.int64)
            finally:state.close()
            return rows

        floating=generate('floating');tokens=[row['greedyToken'] for row in floating]
        report['floating'].update(outputTokens=tokens,outputText=flash_tokenizer.decode(args.source,tokens,len(tokens)))
        publish(report_path,report)
        if args.mode=='quantized':
            callback=StreamedQuantized(graph,12)
            quantized=generate('quantized',forced=tokens,callback=callback)
            ordered,inventory=callback.inventory(graph.order)
            (work/'coefficient-inventory.json').write_text(json.dumps(ordered)+'\n')
            errors=[float(np.abs(np.load(work/f'quantized-step{step}-logits.npy')-np.load(work/f'floating-step{step}-logits.npy')).max()) for step in range(args.steps)]
            report['quantized'].update(quantizationBits=12,teacherForcedFromFloating=True,greedyMatchesFloating=[row['greedyToken']==tokens[step] for step,row in enumerate(quantized)],
                maxAbsLogitErrorVersusFloating=errors,coefficientInventorySha256=inventory,coefficientProjections=len(ordered),coefficientCacheWritten=False)
        report.update(completed=True,fullFlashTextModelExecutedInClear=True,elapsedSeconds=round(time.perf_counter()-started,1),
            cooperativeLockWaitSeconds=round(turn.waited,1),peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        publish(report_path,report);log('completed',elapsedSeconds=report['elapsedSeconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--reference');parser.add_argument('--work')
    parser.add_argument('--mode',choices=['metadata','floating','quantized'],default='metadata')
    parser.add_argument('--steps',type=int,default=4);parser.add_argument('--context-limit',type=int,default=64)
    parser.add_argument('--threads',type=int,default=8);parser.add_argument('--lock');parser.add_argument('--pause-file')
    args=parser.parse_args()
    if args.mode=='metadata':print(json.dumps(dict(schema=SCHEMA,metadataOnly=True,**metadata(args.source))))
    else:
        if not args.reference or not args.work or not 0<args.steps<=16:raise ValueError('Explicit reference, work directory and bounded step count required')
        run(args)
