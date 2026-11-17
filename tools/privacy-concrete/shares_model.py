"""Small-Qwen generation with every linear layer delegated to two evaluators on additive shares.

The client keeps the prompt, the embedding table and the cheap non-linear steps. It reads no linear-layer
coefficient while executing. The same process still loads the reference model so the result can be compared
after the route closes; that comparison is research validation, not part of the route.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from full_model import model_step,prepare
from full_status import ExecutionRejected,failure_event
from plan_backends import qwen_backend
from protocol import require_active
from public_qwen_fixtures import PUBLIC_FIXTURES
from shares import MECHANISM,ProcessEvaluator,ShareRoute,build_anchor
from verified_execution import complete_verified_route,VALIDATION_BOUNDARY

PREPARATION_PROFILE='qwen-all-linear-ntt12-check'
cpu_seconds=lambda who:sum(resource.getrusage(who)[:2])


def publish(directory,projections):
    """What the model owner posts beside the weights: where each public matrix is."""
    rows=[dict(name=projection.name,inputWidth=projection.definition['inputWidth'],outputWidth=projection.definition['outputWidth'],
               weightId=projection.definition['weightId'],file=projection.weight_path.name,dtype=np.dtype(projection.encoding['dtype']).str)
          for projection in projections]
    (Path(directory)/'shares-manifest.json').write_text(json.dumps(dict(schema='doppler.two-server-shares-manifest/v1',projections=rows)))


def requester_anchors(seed,projections):
    """One pass over the public weights with the requester's secret probes. Done once per model, not per request."""
    anchors={}
    for projection in projections:
        with projection.coefficient_matrix() as weights:anchors[projection.name]=build_anchor(seed,projection.name,weights)
    return anchors


def run(model_dir,cache_dir,prompt,commands=None,progress=lambda event:None,cancelled=None):
    settings,_,_,_=qwen_backend(PREPARATION_PROFILE)
    started=time.perf_counter()
    model,tokenizer,_,projections,plan=prepare(model_dir,cache_dir,PREPARATION_PROFILE,cancelled=cancelled)
    prepared=time.perf_counter()
    publish(cache_dir,projections)
    published=time.perf_counter()
    probe_seed=os.urandom(32)
    anchors=requester_anchors(probe_seed,projections)
    anchored=time.perf_counter()
    tokens=tokenizer.encode(prompt)
    if not 1<=len(tokens)<=settings['contextBucket']:raise ValueError('Prompt exceeds the fixed prefill bucket')
    original_count=len(tokens)
    local=[sys.executable,str(Path(__file__).with_name('shares_evaluator.py')),'--cache-dir',str(cache_dir)]
    commands=commands or [local,local]
    route=ShareRoute(tuple(ProcessEvaluator(command,label) for command,label in zip(commands,'ab')),probe_seed,anchors,cancelled=cancelled,
                     observe=lambda kind,value:None)
    try:
        for projection in projections:projection.shares=route
        caches={mode:DynamicCache() for mode in ['float-reference','clear-reference','shared']}
        step_ms=[];errors=[];float_matches=[]
        client_cpu=cpu_seconds(resource.RUSAGE_SELF);child_cpu=cpu_seconds(resource.RUSAGE_CHILDREN)
        def execute(step):
            require_active(cancelled)
            logits,duration=model_step(model,projections,caches['shared'],tokens,original_count,step,'shared',settings['contextBucket'],151643)
            step_ms.append(duration);tokens.append(int(logits.argmax()))
            progress(dict(event='step-executed',step=step+1,sharedProjectionCalls=route.calls))
            return logits.detach().clone()
        usage={}
        def validate(step,actual):
            if not usage:
                usage.update(client=cpu_seconds(resource.RUSAGE_SELF)-client_cpu,evaluators=cpu_seconds(resource.RUSAGE_CHILDREN)-child_cpu)
            history=tokens[:original_count+step];logits={}
            for mode in ['float-reference','clear-reference']:
                require_active(cancelled)
                logits[mode],_=model_step(model,projections,caches[mode],history,original_count,step,mode,settings['contextBucket'],151643)
            errors.append(float((actual-logits['clear-reference']).abs().max()))
            next_token=int(actual.argmax())
            if next_token!=int(logits['clear-reference'].argmax()):raise ExecutionRejected('token-reference-mismatch')
            float_matches.append(next_token==int(logits['float-reference'].argmax()))
        with torch.no_grad():complete_verified_route(route,settings['steps'],execute,validate)
        details=route.report()
        linear_bytes=sum(4*projection.definition['inputWidth']*projection.definition['outputWidth'] for projection in projections)
        embedding=model.get_input_embeddings().weight
        other_bytes=4*sum(parameter.numel() for name,parameter in model.named_parameters() if '.remote.' not in name and parameter is not embedding)
        # Per projection the client also keeps weight-derived quantization vectors and the bias.
        metadata_bytes=sum(sum(tensor.numel()*tensor.element_size() for tensor in (projection.scale,projection.zero_point,projection.sum_weights))
                           +(0 if projection.remote is None or projection.remote.bias is None else 4*projection.remote.bias.numel()) for projection in projections)
        wall=sum(step_ms)
        return dict(**details,schema='doppler.two-server-shares-execution/v1',qualified=False,productionBackendAvailable=False,
                    securityQualified=False,nonCollusionEstablished=False,validationBoundary=VALIDATION_BOUNDARY,mechanism=MECHANISM,
                    planId=plan.identity,modelId=plan.definition()['modelId'],linearProjectionCount=len(projections),steps=settings['steps'],
                    evaluatorTransports=['ssh' if Path(command[0]).name=='ssh' else 'local-process' for command in commands],
                    clientLocalOutputTokens=tokens[original_count:],clientLocalOutputText=tokenizer.decode(tokens[original_count:]),
                    matchesFullQuantizedGreedyReference=True,floatGreedyMatches=float_matches,maxLogitErrors=errors,
                    stepMs=step_ms,executionWallMs=wall,
                    clientNonLinearAndOrchestrationMs=wall-details['clientShareMs']-details['evaluatorRoundTripMs'],
                    clientCpuSeconds=usage.get('client'),localEvaluatorCpuSeconds=usage.get('evaluators'),
                    clientStoredBytes=dict(embeddingTable=4*embedding.numel(),normsAndOtherParameters=other_bytes,quantizationVectorsAndBiases=metadata_bytes,
                                           anchors=sum(anchor.nbytes for anchor in anchors.values())),
                    linearWeightBytesHeldByEvaluatorsFp32=linear_bytes,clientReadsLinearCoefficientsDuringExecution=False,
                    referenceWeightsInThisProcess='Loaded for the post-close comparison only; the route itself never reads them.',
                    preparationMs=(prepared-started)*1000,ownerPublicationMs=(published-prepared)*1000,requesterAnchorPassMs=(anchored-published)*1000,
                    clientPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
    finally:
        route.close()
        for projection in projections:projection.shares=None


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--public-fixture',choices=tuple(PUBLIC_FIXTURES),required=True)
    parser.add_argument('--evaluator-command',action='append',help='JSON array; give twice to place the evaluators elsewhere')
    args=parser.parse_args()
    commands=[json.loads(item) for item in args.evaluator_command] if args.evaluator_command else None
    if commands is not None and len(commands)!=2:raise ValueError('Exactly two evaluator commands required')
    began=time.perf_counter()
    try:
        report=run(args.model_dir,args.cache_dir,PUBLIC_FIXTURES[args.public_fixture],commands,
                   progress=lambda event:print(json.dumps(event),file=sys.stderr,flush=True))
    except BaseException as error:
        print(json.dumps(failure_event(error,(time.perf_counter()-began)*1000)),file=sys.stderr,flush=True)
        raise
    report['publicFixtureId']=args.public_fixture
    print(json.dumps(report,indent=2))
