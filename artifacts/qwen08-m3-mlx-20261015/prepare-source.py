import hashlib,json,pathlib,urllib.request
root=pathlib.Path(__file__).parent
revision='2fc06364715b967f1860aea9cf38778875588b17'
repo='Qwen/Qwen3.5-0.8B'
with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=30) as response: metadata=json.load(response)
assert metadata['sha']==revision
(root/'source-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
folder=pathlib.Path('.cache/qwen08-source');folder.mkdir(parents=True,exist_ok=True)
receipts={}
for entry in metadata['siblings']:
 name=entry['rfilename']
 if not name.endswith(('.json','.safetensors','.txt','.jinja')):continue
 if 'preprocessor' in name:continue
 target=folder/name; expected=entry.get('lfs',{}).get('sha256')
 if target.exists() and expected:
  with target.open('rb') as f: existing=hashlib.file_digest(f,'sha256').hexdigest()
  if existing==expected:receipts[name]=existing;continue
 tmp=target.with_suffix(target.suffix+'.download');h=hashlib.sha256();total=0;last=0
 with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',timeout=30) as response,tmp.open('wb') as output:
  while chunk:=response.read(4*1024*1024):
   output.write(chunk);h.update(chunk);total+=len(chunk)
   if total-last>=128*1024*1024: print(name,round(total/2**20),'MiB',flush=True);last=total
 if expected and h.hexdigest()!=expected:raise RuntimeError(f'Hash mismatch: {name}')
 tmp.replace(target);receipts[name]=h.hexdigest();print(name,'verified',total,flush=True)
(root/'source-files.json').write_text(json.dumps(receipts,indent=2)+'\n')
