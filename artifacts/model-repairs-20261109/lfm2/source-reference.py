import json,hashlib,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
root=Path('/run/media/x/models/huggingface_cache/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/e129ea4241f9232d771540bd8ed4753feebaedc2')
torch.set_num_threads(3)
t=AutoTokenizer.from_pretrained(root,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,dtype=torch.float32,device_map='cpu',attn_implementation='eager').eval()
rows=[]
for prompt in ['What color is the sky on a clear day? Answer in one sentence.','What is 2 + 2? Answer with just the number.','Write a Python function that returns the larger of two numbers.']:
 text=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True);x=t(text,return_tensors='pt',add_special_tokens=False)
 with torch.inference_mode():
  first=m(**x,use_cache=False,output_hidden_states=True);top=torch.topk(first.logits[0,-1].float(),8);y=m.generate(**x,max_new_tokens=24,do_sample=False,use_cache=True)
 ids=y[0,x.input_ids.shape[1]:].tolist();rows.append({'prompt':prompt,'formattedPrompt':text,'promptTokenIds':x.input_ids[0].tolist(),'generatedTokenIds':ids,'generatedTokens':len(ids),'output':t.decode(ids),'firstLogitsTop':{'ids':top.indices.tolist(),'values':top.values.tolist()},'firstLayerLastSlice':first.hidden_states[1][0,-1,:16].tolist()});print(json.dumps(rows[-1]),flush=True)
import transformers
Path('/tmp/doppler-lfm2-source-reference.json').write_text(json.dumps({'model':'LiquidAI/LFM2.5-1.2B-Instruct','revision':root.name,'sourceSha256':hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest(),'execution':{'device':'cpu','dtype':'float32','sampling':'greedy-argmax-f32-logits','transformers':transformers.__version__},'rows':rows},indent=2))
