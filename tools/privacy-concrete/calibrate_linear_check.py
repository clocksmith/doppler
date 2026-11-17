"""Measure honest decode error of the checked route on real model activations.

Supports assumption A1 in INTEGRITY.md for one model and one public fixture: it
runs the clear integer route, and for a sample of tiles also encrypts the real
quantized activation, evaluates it with the installed native build under a
fresh disposable key, decrypts, and compares with the exact integer product.
Client-local; no evaluator, no private input, no security or production claim.
"""
import argparse
import json
import sys
import time
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
import concrete_ml_extensions as ext
import full_model
from full_plan import ranges
from linear_check import POLICY,POLICY_ID,honest_norm_bound,noise_model,tolerance
from plan_backends import qwen_backend
from public_qwen_fixtures import PUBLIC_FIXTURES


def calibrate(model_dir,cache_dir,profile,fixture,every):
    settings,_,_,_=qwen_backend(profile)
    if settings.get('integrity')!=POLICY['id']:raise ValueError('Checked-integrity profile required')
    noise=noise_model(settings['parameterProfile'])
    model,tokenizer,executor,projections,plan=full_model.prepare(model_dir,cache_dir,profile)
    parameters=executor.glwe_crypto_params
    secret,key=ext.cpu_create_private_key(parameters);ext.cpu_prepare_ntt_key(key)
    geometry=(plan.definition()['inputTileWidth'],plan.definition()['outputTileWidth'])
    current=dict(name=None);calls=dict(count=0);rows=[];exact=full_model.integer_dot
    for projection in projections:
        projection.register_forward_pre_hook(lambda module,arguments:current.update(name=module.name))

    def measured(inputs,weights):
        values=np.ascontiguousarray(inputs.long().numpy(),dtype=np.int64)
        for out_begin,out_end in ranges(weights.shape[1],geometry[1]):
            for in_begin,in_end in ranges(weights.shape[0],geometry[0]):
                calls['count']+=1
                if calls['count']%every:continue
                tile=np.ascontiguousarray(weights[in_begin:in_end,out_begin:out_end],dtype=np.int64)
                part=np.ascontiguousarray(values[:,in_begin:in_end])
                began=time.perf_counter()
                cipher=ext.encrypt_matrix(pkey=secret,crypto_params=parameters,data=part.astype(np.uint64))
                answer=ext.cpu_matrix_multiplication_ntt(cipher,tile.astype(np.uint64),key)
                decoded=ext.decrypt_matrix(answer,secret,parameters,out_end-out_begin).astype(np.int64)
                seconds=time.perf_counter()-began
                product=part@tile;error=(decoded-product).astype(np.float64);width=out_end-out_begin
                norms=np.sqrt((error**2).sum(axis=1));bound=honest_norm_bound(width,noise)
                rows.append(dict(projection=current['name'],inputWidth=in_end-in_begin,outputWidth=width,rows=int(part.shape[0]),
                    maxRowL2=float(norms.max()),bound=float(bound),ratio=float(norms.max()/bound),maxAbsError=float(np.abs(error).max()),
                    meanError=float(error.mean()),signalRms=float(np.sqrt((product.astype(np.float64)**2).mean())),seconds=round(seconds,3)))
                print(json.dumps(dict(event='calibrated-tile',**rows[-1])),file=sys.stderr,flush=True)
        return exact(inputs,weights)

    full_model.integer_dot=measured
    try:
        tokens=tokenizer.encode(PUBLIC_FIXTURES[fixture]);count=len(tokens);cache=DynamicCache()
        with torch.no_grad():
            for step in range(2):
                logits,_=full_model.model_step(model,projections,cache,tokens,count,step,'clear-reference',settings['contextBucket'],151643)
                tokens.append(int(logits.argmax()))
    finally:full_model.integer_dot=exact
    widths={}
    for row in rows:
        entry=widths.setdefault(str(row['outputWidth']),dict(tiles=0,maxRatio=0.0,maxRowL2=0.0,maxAbsError=0.0,minSignalRms=float('inf')))
        entry['tiles']+=1
        for key_name,source in [('maxRatio','ratio'),('maxRowL2','maxRowL2'),('maxAbsError','maxAbsError')]:entry[key_name]=max(entry[key_name],row[source])
        entry['minSignalRms']=min(entry['minSignalRms'],row['signalRms'])
    for width,entry in widths.items():
        entry.update(bound=honest_norm_bound(int(width),noise),toleranceOverWeight=tolerance(int(width),noise)/2**15)
    return dict(schema='doppler.linear-check-model-calibration/v1',backendProfile=profile,parameterProfile=settings['parameterProfile'],
        integrityPolicyId=POLICY_ID,publicFixtureId=fixture,modelId=plan.definition()['modelId'],sampledEvery=every,sampledTiles=len(rows),
        stepsCovered=2,assumptionHeld=all(row['ratio']<=1 for row in rows),maxRatio=max(row['ratio'] for row in rows),byOutputWidth=widths,
        scope='One model, one public fixture, sampled tiles, one host. Not a proof of assumption A1.',securityQualified=False,tiles=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--backend-profile',default='qwen-all-linear-ntt12-check');parser.add_argument('--public-fixture',choices=tuple(PUBLIC_FIXTURES),required=True)
    parser.add_argument('--every',type=int,default=6)
    args=parser.parse_args()
    if args.every<=0:raise ValueError('Positive sampling interval required')
    print(json.dumps(calibrate(args.model_dir,args.cache_dir,args.backend_profile,args.public_fixture,args.every),indent=1))
