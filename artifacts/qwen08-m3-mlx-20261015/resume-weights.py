import concurrent.futures,hashlib,json,os,pathlib,urllib.request
root=pathlib.Path(__file__).parent
policy=json.loads((root/'policy.json').read_text())['source']
metadata=json.loads((root/'source-metadata.json').read_text())
entry=next(e for e in metadata['siblings'] if e['rfilename']==policy['weightsFile'])
size=entry['size'];folder=pathlib.Path('.cache/qwen08-source')
target=folder/policy['weightsFile'];partial=target.with_suffix(target.suffix+'.download')
start=partial.stat().st_size
url=f"https://huggingface.co/{policy['repo']}/resolve/{policy['revision']}/{policy['weightsFile']}"
fd=os.open(partial,os.O_WRONLY)
def fetch(begin):
 end=min(begin+8*1024*1024,size)-1
 req=urllib.request.Request(url,headers={'Range':f'bytes={begin}-{end}'})
 with urllib.request.urlopen(req,timeout=45) as response:
  expected=f'bytes {begin}-{end}/{size}'
  assert response.status==206 and response.headers.get('Content-Range')==expected,(response.status,response.headers.get('Content-Range'),expected)
  data=response.read();assert len(data)==end-begin+1
 assert os.pwrite(fd,data,begin)==len(data)
 return len(data)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  total=start
  for count in pool.map(fetch,range(start,size,8*1024*1024)):
   total+=count;print(f'{total}/{size}',flush=True)
finally:os.close(fd)
with partial.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==policy['weightsSha256']
partial.replace(target);print('Full source SHA-256 verified',flush=True)
