"""Capture actual executed source/model bytes and environment without GPU work."""
import hashlib,json,pathlib,subprocess,platform,datetime
ROOT=pathlib.Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def command(args):
 p=subprocess.run(args,capture_output=True,text=True);return {'argv':args,'exitCode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
runtime=ROOT/'runtime'
files={str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file() and 'node_modules' not in p.parts}
contract=json.loads((ROOT/'contract.json').read_text())
tracked=subprocess.check_output(['git','ls-tree','-r',contract['sourceRevision'],'src','package.json'],cwd=REPO,text=True)
expected={line.split('\t')[1]:line.split()[2] for line in tracked.splitlines() if line.split()[0]=='100644' or line.split()[0]=='100755'}
differences=[]
for rel,blob in expected.items():
 p=runtime/rel
 if not p.exists():differences.append({'path':rel,'reason':'not in executed package'});continue
 data=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if actual!=blob:differences.append({'path':rel,'reason':'bytes differ from declared revision','executedSha256':sha(p)})
model=pathlib.Path('$HOME/deco/hatch/pooled-doppler-bench/_work/doppler-successor')
model_files={str(p.relative_to(model)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in model.rglob('*') if p.is_file()}
result={'schema':'doppler.amd-host-baseline-identity.v1','capturedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':platform.node(),'sourceRevision':contract['sourceRevision'],'executedFilesSha256':files,'sourceRevisionDifferences':differences,'modelPath':str(model),'modelFiles':model_files,'environment':[command(a) for a in [['uname','-a'],['node','--version'],['/opt/google/chrome/chrome','--version'],['dpkg-query','-W','mesa-vulkan-drivers','libvulkan1'],['free','-b'],['df','-B1',str(REPO)],['git','-C',str(REPO),'status','--short']]],'scope':'Qwen3.5 0.8B only; warm OPFS. All timed samples retain exact inputs, outputs, resolved runtime and memory allocation counters; no physical resident-memory measurement.'}
(ROOT/'baseline-identity.json').write_text(json.dumps(result,indent=2)+'\n')
print('Captured',len(files),'source files;',len(model_files),'model files;',len(differences),'revision differences')
