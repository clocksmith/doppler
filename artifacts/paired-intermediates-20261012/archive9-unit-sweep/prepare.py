from pathlib import Path
import hashlib,json,os,subprocess,tarfile,tempfile,datetime
REV='bddea7ef32ce54e0dea14793258a6857db65aad7'
ARCHIVE=Path('/var/tmp/doppler-gpu-0.6.27.tgz')
EXPORT=Path('/var/tmp/doppler-archive9-cpu-source.tar')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(EXPORT.read_bytes())=='ac35b8c4bc718739e5e81bcff037b2eab495de43fb35af35b2a6b74510252436'
assert sha(ARCHIVE.read_bytes())=='8cae56b0748a5c66add985122d8bbf72b7492e2e0828f713f1b0d7425a59a4e5'
base=Path(tempfile.mkdtemp(prefix='doppler-archive9-unit-',dir='/var/tmp'))
source=base/'workspace/doppler'; source.mkdir(parents=True)
evidence=base/'evidence';evidence.mkdir()
with tarfile.open(EXPORT) as t:t.extractall(source,filter='data')
archive=base/'archive-9.tgz';archive.write_bytes(ARCHIVE.read_bytes())
rows=[]
with tarfile.open(archive) as t:
 for member in t:
  if not member.isfile():continue
  p=Path(member.name); assert p.parts[0]=='package' and '..' not in p.parts
  rel=str(Path(*p.parts[1:])); data=t.extractfile(member).read(); actual=(source/rel).read_bytes()
  assert data==actual,rel
  rows.append({'path':rel,'sha256':sha(data),'bytes':len(data)})
(evidence/'archive-files.json').write_text(json.dumps(rows,indent=2)+'\n')
env=dict(os.environ)
for k in list(env):
 if k.startswith('GIT_'):del env[k]
gitdir=evidence/'git-metadata'
subprocess.run(['git','init','--bare',str(gitdir)],check=True,env=env,stdout=subprocess.PIPE)
objects=subprocess.check_output(['git','-C','$HOME/deco/doppler','rev-parse','--git-common-dir'],text=True).strip()+'/objects'
(gitdir/'objects/info/alternates').write_text(objects+'\n')
git=['git','--git-dir='+str(gitdir)]
if subprocess.run(git+['cat-file','-e',REV],env=env,stderr=subprocess.DEVNULL).returncode:
 subprocess.run(git+['fetch','--no-tags','/var/tmp/doppler-archive9-unit-metadata.bundle','main'],check=True,env=env)
for key,value in [('core.bare','false'),('core.worktree',str(source)),('user.name','Doppler Frozen CPU Sweep'),('user.email','frozen-sweep@invalid')]:
 subprocess.run(git+['config',key,value],check=True,env=env)
(gitdir/'HEAD').write_text(REV+'\n')
(source/'.git').write_text('gitdir: '+str(gitdir)+'\n')
subprocess.run(git+['read-tree',REV],check=True,env=env)
deps=Path('$HOME/deco/doppler/node_modules');(source/'node_modules').mkdir()
for p in deps.iterdir():(source/'node_modules'/p.name).symlink_to(p,target_is_directory=p.is_dir())
sibling=base/'workspace/reploid';siblingrev='99ca0456d01e8f08e070ec998de8834e4b6008f7';siblings=[]
for name in ['tests/fixtures/distributed-reference-model.js','tests/fixtures/distributed-reference-manifest.json.gz']:
 b=subprocess.check_output(['git','-C','$HOME/deco/reploid','show',siblingrev+':'+name]);p=sibling/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);siblings.append({'path':name,'sha256':sha(b)})
(evidence/'sibling-binding.json').write_text(json.dumps({'sourceRevision':siblingrev,'files':siblings},indent=2)+'\n')
tracked=subprocess.check_output(['git','-C',str(source),'ls-files','-z'],env=env).decode().split('\0')
allfiles={p:sha((source/p).read_bytes()) for p in tracked if p and (source/p).is_file()}
(evidence/'tracked-before.json').write_text(json.dumps(allfiles,indent=2)+'\n')
assert not subprocess.check_output(['git','-C',str(source),'diff','--name-only',REV],env=env).strip()
binding={'sourceCommit':REV,'snapshot':str(source),'evidence':str(evidence),'archive':str(archive),'archiveSha256':sha(archive.read_bytes()),'sourceExportSha256':sha(EXPORT.read_bytes()),'archiveFileCount':len(rows),'archiveDifferences':[],'trackedFileCount':len(allfiles),'snapshotPackageLockSha256':sha((source/'package-lock.json').read_bytes()),'dependencyDirectory':str(deps),'installedDependencyLockSha256':sha((deps/'.package-lock.json').read_bytes()),'node':subprocess.check_output(['node','--version'],text=True).strip(),'privateGitDirectory':str(gitdir),'gitObjectAlternate':objects,'noInheritedGitVariables':True,'affinity':[4,5,6,7],'nice':10,'env':{'DOPPLER_NODE_WEBGPU_MODULE':str(evidence/'intentionally-absent-webgpu.js'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},'command':['taskset','-c','4-7','nice','-n','10','node','tools/run-node-tests.js','--suite','unit'],'runnerFileConcurrency':1,'scope':'Frozen-source complete CPU unit sweep; optional GPU unavailable, no inference or physical acceptance.'}
(evidence/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
print(json.dumps(binding))
