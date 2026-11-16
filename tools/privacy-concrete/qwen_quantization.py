"""Actual full Qwen int8-versus-float quality fixtures; no encrypted run is repeated."""
import argparse
import json
import os
import resource
import platform
import time
from unittest.mock import patch
import torch
from transformers.cache_utils import DynamicCache
import concrete_ml_extensions as ext
from full_model import prepare,model_step
from full_plan import implementation_versions
from full_status import failure_event

from public_qwen_fixtures import SPEC as FIXTURES


def forbidden(*args,**kwargs):raise AssertionError('Clear quantization check attempted encrypted/key execution')


def run(model_dir,cache_dir):
    if os.environ.get('RAYON_NUM_THREADS')!='4':raise ValueError('Explicit installed Qwen CPU budget required')
    # Fail mechanically if any reference path unexpectedly generates keys or encrypts.
    with patch.object(ext,'cpu_create_private_key',side_effect=forbidden),patch.object(ext,'encrypt_matrix',side_effect=forbidden):
        began=time.perf_counter();model,tokenizer,executor,projections,plan=prepare(model_dir,cache_dir)
        encoded=[tokenizer.encode(case['prompt']) for case in FIXTURES['fixtures']]
        if any(not 1<=len(tokens)<=FIXTURES['prefillRows'] for tokens in encoded):raise ValueError('Public quality fixture exceeds its declared prefill bucket')
        results=[]
        with torch.no_grad():
            for case,prefix in zip(FIXTURES['fixtures'],encoded):
                quantized=list(prefix);floating=list(prefix);count=len(prefix)
                caches={mode:DynamicCache() for mode in ['quantized','float-teacher','float-independent']}
                errors=[];cosines=[];teacher_matches=[];durations=[]
                for step in range(FIXTURES['steps']):
                    q,ms=model_step(model,projections,caches['quantized'],quantized,count,step,'clear-reference',FIXTURES['prefillRows'],FIXTURES['padTokenId']);durations.append(ms)
                    f,_=model_step(model,projections,caches['float-teacher'],quantized,count,step,'float-reference',FIXTURES['prefillRows'],FIXTURES['padTokenId'])
                    independent,_=model_step(model,projections,caches['float-independent'],floating,count,step,'float-reference',FIXTURES['prefillRows'],FIXTURES['padTokenId'])
                    if not all(torch.isfinite(value).all() for value in [q,f,independent]):raise ValueError('Finite reference logits required')
                    errors.append(float((q-f).abs().max()));cosines.append(float(torch.nn.functional.cosine_similarity(q.unsqueeze(0),f.unsqueeze(0)).item()))
                    token=int(q.argmax());teacher_matches.append(token==int(f.argmax()))
                    quantized.append(token);floating.append(int(independent.argmax()))
                results.append(dict(fixtureId=case['id'],publicPrompt=case['prompt'],inputTokens=count,
                    quantizedOutputTokens=quantized[count:],floatingOutputTokens=floating[count:],
                    quantizedOutputText=tokenizer.decode(quantized[count:]),floatingOutputText=tokenizer.decode(floating[count:]),
                    independentGreedySequenceMatches=quantized[count:]==floating[count:],teacherForcedGreedyMatches=teacher_matches,
                    maxTeacherForcedLogitErrors=errors,teacherForcedLogitCosines=cosines,quantizedStepMs=durations))
        return dict(schema='doppler.full-qwen-public-quantization-check/v1',model=FIXTURES['model'],modelId=plan.definition()['modelId'],planId=plan.identity,
                    linearModules=len(projections),fixtureCount=len(results),stepsPerFixture=FIXTURES['steps'],encryptedExecution=False,
                    keyGenerationAllowed=False,productionBackendAvailable=False,securityQualified=False,
                    implementationVersions=implementation_versions(),results=results,elapsedMs=(time.perf_counter()-began)*1000,
                    clientLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                    interpretation='Fixed public prompt quality checks only; not a corpus benchmark or encrypted execution evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',required=True);parser.add_argument('--cache-dir',required=True);args=parser.parse_args();began=time.perf_counter()
    try:report=run(args.model_dir,args.cache_dir)
    except BaseException as error:
        print(json.dumps(dict(schema='doppler.full-qwen-public-quantization-check/v1',productionBackendAvailable=False,encryptedExecution=False,status=failure_event(error,(time.perf_counter()-began)*1000))),flush=True);raise
    print(json.dumps(report,indent=2))
