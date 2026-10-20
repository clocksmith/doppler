import hashlib,json,pathlib,sys,time
root,index,out=map(pathlib.Path,sys.argv[1:]);index_bytes=index.read_bytes();pieces=json.loads(index_bytes)
raw=(root/'manifest.json').read_bytes();manifest=json.loads(raw)
receipt={'modelId':manifest['modelId'],'manifestSha256':hashlib.sha256(raw).hexdigest(),'pieceIndexSha256':hashlib.sha256(index_bytes).hexdigest(),
 'authority':'Existing Reploid SHA-256 piece index, independent of legacy manifest BLAKE3 fields','started':time.time(),'files':[],'passed':False}
try:
 assert pieces['manifestIdentity']=='sha256:'+receipt['manifestSha256']
 expected={s['filename'] for s in manifest['shards']}|{manifest['tokenizer']['file']}
 for entry in pieces['files']:
  file=root/entry['path'];before=file.stat();data=file.read_bytes();cursor=0
  assert file.stat().st_mtime_ns==before.st_mtime_ns
  assert len(data)==entry['size'],entry['path']
  for piece in entry['pieces']:
   assert piece['offset']==cursor
   block=data[cursor:cursor+piece['size']]
   assert 'sha256:'+hashlib.sha256(block).hexdigest()==piece['identity'],entry['path']
   cursor+=piece['size']
  assert cursor==len(data)
  receipt['files'].append({'file':entry['path'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'pieces':len(entry['pieces']),'mtimeNs':before.st_mtime_ns})
  expected.discard(entry['path'])
 assert not expected,sorted(expected)
 receipt['passed']=True
except Exception as error:receipt['failure']=repr(error);raise
finally:receipt['elapsedSeconds']=time.time()-receipt['started'];out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':receipt['passed'],'files':len(receipt['files']),'elapsedSeconds':receipt['elapsedSeconds']}))
