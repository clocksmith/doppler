import json,pathlib,hashlib,torch,numpy as np
from safetensors import safe_open
root=pathlib.Path('/home/x/deco/doppler/models/local/qwen-3-5-0-8b-q4k-ehaf16');m=json.loads((root/'manifest.json').read_text());key='model.language_model.embed_tokens.weight';e=m['tensors'][key]
data=b''.join((root/m['shards'][s['shardIndex']]['filename']).read_bytes()[s['offset']:s['offset']+s['size']] for s in e['spans']);actual=np.frombuffer(data,dtype='<f2').reshape(e['shape']);assert len(data)==e['size']
original=pathlib.Path('/var/tmp/doppler-qwen35-08b-source-20261012/model.safetensors-00001-of-00001.safetensors')
with safe_open(original,framework='pt',device='cpu') as f:source=f.get_tensor(key)
count=0;examples=[];max_error=0;sum_error=0;expected_hash=hashlib.sha256();source_min=float(source.min());source_max=float(source.max());nonfinite=0
for start in range(0,len(actual),4096):
 expected=source[start:start+4096].float().numpy();got=actual[start:start+4096].astype(np.float32);expected_hash.update(expected.astype('<f2').tobytes());bad=got!=expected;count+=int(bad.sum());nonfinite+=int((~np.isfinite(got)).sum());diff=got-expected;max_error=max(max_error,float(np.abs(diff).max()));sum_error+=float(np.square(diff.astype(np.float64)).sum())
 if len(examples)<20:
  for i,j in np.argwhere(bad)[:20-len(examples)]:examples.append({'row':int(start+i),'column':int(j),'source':float(expected[i,j]),'deployed':float(got[i,j])})
r={'dtype':str(source.dtype),'shape':e['shape'],'sourceMin':source_min,'sourceMax':source_max,'mismatchCount':count,'elementCount':actual.size,'maxAbsoluteError':max_error,'rmse':(sum_error/actual.size)**.5,'nonfiniteDeployed':nonfinite,'expectedF16Sha256':expected_hash.hexdigest(),'deployedF16Sha256':hashlib.sha256(data).hexdigest(),'examples':examples}
pathlib.Path('/tmp/embedding-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
