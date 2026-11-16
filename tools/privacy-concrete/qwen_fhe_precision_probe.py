"""One actual public first-Q projection under two declared parameter sets; not full-model evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import concrete_ml_extensions as ext
from transformers.cache_utils import DynamicCache
from full_model import prepare,model_step
from integer_reference import integer_dot
from protocol import POLICY


def run(model_dir,cache_dir):
    model,tokenizer,executor,projections,plan=prepare(model_dir,cache_dir,backend_profile='qwen-all-linear-ntt-48')
    first=projections[0];captured=[]
    handle=first.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    tokens=tokenizer.encode('The capital of France is')
    try:
        with torch.no_grad():model_step(model,projections,DynamicCache(),tokens,len(tokens),0,'float-reference',8,151643)
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('Exact first projection diagnostic required')
    q,scale,zero,shape=executor._dynamic_input_quantization(captured[0],first.context,False)
    weights=np.array(first.weights,dtype=np.int64);expected=integer_dot(q,first.weights)
    clear=executor._apply_correction_and_dequantize(expected,q,zero,first.zero_point,first.sum_weights,first.definition['inputWidth'],scale,first.scale)
    high=json.loads(Path(__file__).with_name('precision-policy.json').read_text());results=[]
    for name,definition in [('default',POLICY['parameters']),('glwe-precision-48-level4/v1',high['parameters'])]:
        params=ext.MatmulCryptoParameters.deserialize(json.dumps(definition));secret,key=ext.cpu_create_private_key(params);ext.cpu_prepare_ntt_key(key)
        encrypted=ext.encrypt_matrix(pkey=secret,crypto_params=params,data=np.ascontiguousarray(q.long().numpy(),dtype=np.uint64))
        result=ext.cpu_matrix_multiplication_ntt(encrypted,weights.astype(np.uint64),key)
        decoded=ext.decrypt_matrix(result,secret,params,weights.shape[1]).astype(np.int64)
        actual=executor._apply_correction_and_dequantize(torch.from_numpy(decoded),q,zero,first.zero_point,first.sum_weights,weights.shape[0],scale,first.scale)
        results.append(dict(parameterProfile=name,rows=q.shape[0],inputWidth=weights.shape[0],outputWidth=weights.shape[1],maxIntegerError=int(np.abs(decoded.astype(object)-expected.numpy().astype(object)).max()),
            maxDequantizedError=float((actual-clear).abs().max()),evaluationKeyBytes=len(key.serialize())))
        secret=key=None
    return dict(schema='doppler.public-qwen-same-input-fhe-precision-probe/v1',fixtureId='qwen05b-capital',modelId=plan.definition()['modelId'],projection=first.name,
                productionBackendAvailable=False,securityQualified=False,fullModelExecuted=False,results=results,
                interpretation='Disposable public-fixture keys, same actual first-Q input/weights; not production decryption diagnostics or proof of the third-step failure cause')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--cache-dir',required=True);a=p.parse_args();print(json.dumps(run(a.model_dir,a.cache_dir),indent=2))
