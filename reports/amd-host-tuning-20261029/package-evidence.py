from pathlib import Path
import tarfile,hashlib,json,sys
r=Path(sys.argv[1]);out=r/'evidence-package';out.mkdir(exist_ok=True)
files=[];logit=[]
for p in sorted(r.iterdir()):
 if p.is_dir():
  if p.name.startswith('full-logits-'):logit.append(p)
  elif p.name.startswith('screen') or p.name in ['product-screen','product-repeat','product-register-repeat','cpu-observed']:
   files.extend(f for f in p.rglob('*') if f.is_file())
 elif p.suffix in ['.js','.py','.json','.log','.txt','.cpuprofile'] or p.name.endswith('-delta.tar.gz'):
  files.append(p)
manifest=[]
def pack(name, items):
 p=out/name
 with tarfile.open(p,'w:xz',preset=6) as t:
  for f in sorted(items):
   t.add(f,arcname=str(f.relative_to(r)),recursive=False)
   manifest.append({'path':str(f.relative_to(r)),'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'archive':name})
 print(name,p.stat().st_size,flush=True)
 assert p.stat().st_size<95_000_000, 'Split large evidence archives before publication'
pack('raw-runs.tar.xz',files)
group=[];size=0;index=0
for directory in logit:
 items=[f for f in directory.rglob('*') if f.is_file()];total=sum(f.stat().st_size for f in items)
 if size+total>90_000_000 and group:
  pack(f'raw-logits-{index:02d}.tar.xz',group);index+=1;group=[];size=0
 group.extend(items);size+=total
if group:pack(f'raw-logits-{index:02d}.tar.xz',group)
(out/'raw-manifest.json').write_text(json.dumps({'schema':'doppler.tuning-evidence-manifest.v1','entries':manifest},indent=2)+'\n')
print('retained files',len(manifest),flush=True)
