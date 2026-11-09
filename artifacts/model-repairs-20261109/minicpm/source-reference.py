import json,torch,hashlib
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
p='/tmp/doppler-repairs-models/minicpm-source'
torch.set_num_threads(3)
t=AutoTokenizer.from_pretrained(p,trust_remote_code=True,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(p,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
prompt='Q: What color is the sky on a clear day? A:'
text=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
x=t(text,return_tensors='pt'); print('TOKENS',x.input_ids.tolist(),flush=True)
with torch.inference_mode():
 r=m(**x,output_hidden_states=True,use_cache=False);logits=r.logits[0,-1].float();top=torch.topk(logits,8);h=r.hidden_states
 import numpy as np
 np.save('/tmp/doppler-minicpm-source-embedding.npy',h[0][0].float().numpy());np.save('/tmp/doppler-minicpm-source-layer0.npy',h[1][0].float().numpy());np.save('/tmp/doppler-minicpm-source-logits.npy',logits.numpy())
 result={'prompt':prompt,'formattedPrompt':text,'inputIds':x.input_ids[0].tolist(),'logitsTop':{'ids':top.indices.tolist(),'values':top.values.tolist()},'boundarySlices':{'embedding':h[0][0,-1,:16].tolist(),'layer0':h[1][0,-1,:16].tolist()},'sourceSha256':hashlib.sha256(Path(p,'model.safetensors').read_bytes()).hexdigest()}
 y=m.generate(**x,max_new_tokens=16,do_sample=False,use_cache=False); result['generatedIds']=y[0,x.input_ids.shape[1]:].tolist();result['output']=t.decode(result['generatedIds'])
 Path('/tmp/doppler-minicpm-reference.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
