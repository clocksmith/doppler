"""One public first-boundary probe plus clear-only 12-bit full-model control."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from transformers.cache_utils import DynamicCache
import concrete_ml_extensions as ext
from full_model import prepare,model_step
from qwen_quantization import FIXTURES,forbidden
from wide_reference import prepare_wide

POLICY=dict(id='doppler.public-qwen-first-quantization-boundary/v1',exactInputAbsoluteTolerance=0.0,projectionAbsoluteTolerance=.001,projectionRelativeTolerance=.01)


def tensor_record(value):
    value=value.detach().cpu().float().contiguous();flat=value.reshape(-1)
    if not torch.isfinite(value).all():raise ValueError('Finite public boundary values required')
    return dict(shape=list(value.shape),dtype='float32',sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                firstCoordinates=list(range(min(8,flat.numel()))),firstValues=flat[:8].tolist(),minimum=float(flat.min()),maximum=float(flat.max()),mean=float(flat.mean()),rms=float(flat.square().mean().sqrt()))


def run(model_dir,cache_dir):
    with patch.object(ext,'cpu_create_private_key',side_effect=forbidden),patch.object(ext,'encrypt_matrix',side_effect=forbidden):
        began=time.perf_counter();model,tokenizer,executor,projections,plan=prepare(model_dir,cache_dir)
        case=next(c for c in FIXTURES['fixtures'] if c['id']=='sky');tokens=tokenizer.encode(case['prompt']);captures={};active=[None];handles=[]
        def capture(name):
            def observed(module,inputs,output):
                if active[0] is not None:captures[active[0]][name]=output.detach().clone()
            return observed
        for name,module in [('embedding',model.model.embed_tokens),('input-norm',model.model.layers[0].input_layernorm),('q-output',projections[0])]:handles.append(module.register_forward_hook(capture(name)))
        def q_input(module,inputs):
            if active[0] is not None:captures[active[0]]['q-input']=inputs[0].detach().clone()
        handles.append(projections[0].register_forward_pre_hook(q_input))
        try:
            with torch.no_grad():
                for mode in ['float-reference','clear-reference']:
                    captures[mode]={};active[0]=mode
                    logits,_=model_step(model,projections,DynamicCache(),tokens,len(tokens),0,mode,8,151643)
                    captures[mode]['logits']=logits.detach().clone()
        finally:
            for handle in handles:handle.remove()
        boundaries=[]
        for name in ['embedding','input-norm','q-input','q-output','logits']:
            left=captures['float-reference'][name];right=captures['clear-reference'][name]
            boundaries.append(dict(name=name,float=tensor_record(left),quantized=tensor_record(right),maxAbsoluteError=float((left-right).abs().max()),
                exactEqual=torch.equal(left,right),withinProjectionTolerance=torch.allclose(left,right,rtol=POLICY['projectionRelativeTolerance'],atol=POLICY['projectionAbsoluteTolerance'])))
        probe=dict(fixtureId=case['id'],publicPrompt=case['prompt'],inputTokenIds=tokens,policy=POLICY,modelId=plan.definition()['modelId'],boundaries=boundaries)
        del captures,projections,executor,model,tokenizer,plan;gc.collect()
        model,tokenizer,projections,model_id=prepare_wide(model_dir,cache_dir);results=[]
        with torch.no_grad():
            for fixture in FIXTURES['fixtures']:
                prefix=tokenizer.encode(fixture['prompt']);quantized=list(prefix);floating=list(prefix);count=len(prefix)
                if not 1<=count<=8:raise ValueError('Public control fixture exceeds prefill bucket')
                caches={mode:DynamicCache() for mode in ['q','f','teacher']};errors=[];matches=[]
                for step in range(4):
                    q,_=model_step(model,projections,caches['q'],quantized,count,step,'clear-reference',8,151643)
                    teacher,_=model_step(model,projections,caches['teacher'],quantized,count,step,'float-reference',8,151643)
                    f,_=model_step(model,projections,caches['f'],floating,count,step,'float-reference',8,151643)
                    if not all(torch.isfinite(v).all() for v in [q,teacher,f]):raise ValueError('Finite control logits required')
                    errors.append(float((q-teacher).abs().max()));matches.append(int(q.argmax())==int(teacher.argmax()))
                    quantized.append(int(q.argmax()));floating.append(int(f.argmax()))
                results.append(dict(fixtureId=fixture['id'],quantizedOutputTokens=quantized[count:],floatingOutputTokens=floating[count:],
                    quantizedOutputText=tokenizer.decode(quantized[count:]),floatingOutputText=tokenizer.decode(floating[count:]),
                    independentGreedySequenceMatches=quantized[count:]==floating[count:],teacherForcedGreedyMatches=matches,maxTeacherForcedLogitErrors=errors))
        return dict(schema='doppler.public-qwen-precision-control/v1',productionBackendAvailable=False,securityQualified=False,encryptedExecution=False,
                    probe=probe,wideControl=dict(arithmetic='clear-int64-quantization12/v1',modelId=model_id,linearModules=len(projections),results=results),
                    elapsedMs=(time.perf_counter()-began)*1000,interpretation='Public source-precision boundary and clear-only quality control; 12-bit encrypted execution is unavailable')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',required=True);parser.add_argument('--cache-dir',required=True);args=parser.parse_args()
    print(json.dumps(run(args.model_dir,args.cache_dir),indent=2))
