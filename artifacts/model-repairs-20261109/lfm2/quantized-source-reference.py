import json, hashlib, argparse, importlib.metadata
from pathlib import Path
import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--cases',required=True);p.add_argument('--out',required=True);args=p.parse_args()
torch.set_num_threads(3)
root=Path(args.model); source=Path(args.source); raw=(root/'manifest.json').read_bytes(); m=json.loads(raw)
config=AutoConfig.from_pretrained(source,local_files_only=True);config._attn_implementation='eager';config.torch_dtype=torch.float32
with torch.device('meta'): model=AutoModelForCausalLM.from_config(config)
# Rotary frequencies are non-persistent and must be rebuilt on CPU.
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding
for name,module in list(model.named_modules()):
 if isinstance(module,Lfm2RotaryEmbedding):
  parent,key=name.rsplit('.',1);setattr(model.get_submodule(parent),key,Lfm2RotaryEmbedding(config,device='cpu'))
shards={}; identities=[]
for i,s in enumerate(m['shards']):
 data=(root/s['filename']).read_bytes();assert len(data)==s['size'];assert hashlib.sha256(data).hexdigest()==s['hash'];shards[i]=data;identities.append({'file':s['filename'],'sha256':s['hash'],'bytes':len(data)})
wanted=model.state_dict();weights={}
for name,d in m['tensors'].items():
 assert name in wanted and list(wanted[name].shape)==d['shape'],name
 spans=d.get('spans') or [{'shardIndex':d['shard'],'offset':d['offset'],'size':d['size']}]
 data=b''.join(shards[x['shardIndex']][x['offset']:x['offset']+x['size']] for x in spans);assert len(data)==d['size']
 if d['dtype']=='Q4_K_M':
  a=dequantize(np.frombuffer(data,dtype=np.uint8).copy(),GGMLQuantizationType.Q4_K)
  shape=d['shape']; n=int(np.prod(shape))
  if len(shape)==2 or d.get('layout')=='row':
   k=shape[-1];a=a.reshape(-1,((k+255)//256)*256)[:,:k].reshape(-1)
  else:a=a[:n]
  # Diagnostic materialization reference: loaded projections and conv filters use F16.
  a=a.astype(np.float16).astype(np.float32)
 else:
  assert d['dtype'] in ['F16','F32'],d['dtype'];a=np.frombuffer(data,dtype='<f2' if d['dtype']=='F16' else '<f4').astype(np.float32)
 weights[name]=torch.from_numpy(a.reshape(d['shape']).copy())
weights['lm_head.weight']=weights['model.embed_tokens.weight'];model.load_state_dict(weights,strict=True,assign=True);model.tie_weights();model.eval();del weights,shards
rows=[]; t=AutoTokenizer.from_pretrained(source,local_files_only=True)
for row in json.loads(Path(args.cases).read_text())['rows']:
 x=torch.tensor([row['promptTokenIds']],dtype=torch.long)
 with torch.inference_mode(): y=model.generate(x,max_new_tokens=24,do_sample=False,use_cache=True)
 ids=y[0,x.shape[1]:].tolist();r={'prompt':row['prompt'],'promptTokenIds':row['promptTokenIds'],'generatedTokenIds':ids,'output':t.decode(ids)};rows.append(r);print(json.dumps(r),flush=True)
report={'schema':'doppler.quantized-source-reference/v1','qualified':False,'arithmetic':'Source LFM2 F32 equations with independently gguf-decoded Q4_K weights rounded through F16, F32 cache; not bit-exact GPU arithmetic','manifestSha256':hashlib.sha256(raw).hexdigest(),'shards':identities,'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','gguf','numpy']},'rows':rows}
Path(args.out).write_text(json.dumps(report,indent=2))
