"""Prepare a separately identified scalar-softplus candidate; never mutate frozen evidence.

The candidate reuses the pinned weights, source reference and runtime configuration.
Only scalar softplus bodies and their declared shader identities change.
"""
from pathlib import Path
import json, shutil, hashlib, re

import sys
if len(sys.argv)!=7:
    raise SystemExit('Usage: prepare-softplus-candidate.py FROZEN_RUNTIME BASE_RUN REPLACEMENT_SHADER PHYSICAL_TEST NEW_RUNTIME NEW_DATA_DIRECTORY')
old,base,replacement_source,test_source,new,root=map(lambda p:Path(p).resolve(),sys.argv[1:])
assert old.is_dir() and (old/'node_modules').is_dir()
assert all(p.is_file() for p in [base,replacement_source,test_source])
root.mkdir()
new.mkdir()
for name in ['src','tests']:
    shutil.copytree(old/name,new/name,symlinks=True)
for name in ['package.json','package-lock.json','AGENTS.md','CATSCAN.md','GOALS.md']:
    if (old/name).is_file(): shutil.copy2(old/name,new/name)
(new/'node_modules').symlink_to(old/'node_modules',target_is_directory=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
replacement_text=replacement_source.read_text()
start=replacement_text.index('fn softplus(')
end=replacement_text.index('\n}',start)+2
replacement=replacement_text[start:end]
changes=[]
digests=new/'src/config/kernels/kernel-ref-digests.js'
digest_text=digests.read_text()
digest_changes={}
for p in sorted((new/'src/gpu/kernels').glob('*.wgsl')):
    original=p.read_bytes();s=original.decode()
    if 'fn softplus(' not in s: continue
    start=s.index('fn softplus(');end=s.index('\n}',start)+2
    assert 'return log_refined(1.0 + exp_refined(x));' in s[start:end],str(p)
    s=s[:start]+replacement+s[end:];p.write_text(s)
    changes.append({'file':str(p.relative_to(new)),'beforeSha256':sha(original),'afterSha256':sha(p.read_bytes())})
    for m in list(re.finditer(r'"('+re.escape(p.name)+r'#[^"]+)": "([a-f0-9]{64})"',digest_text)):
        key,before=m.groups();entry=key.split('#')[1]
        after=sha((s.replace('\r\n','\n')+'\n@@entry:'+entry).encode())
        assert before==sha((original.decode().replace('\r\n','\n')+'\n@@entry:'+entry).encode())
        digest_text=digest_text.replace('"'+key+'": "'+before+'"','"'+key+'": "'+after+'"')
        digest_changes[key]={'before':before,'after':after}
assert len(changes)==17,len(changes)
digests.write_text(digest_text)
shutil.copy2(test_source,new/'tests/kernels/gated-delta-softplus-physical.test.js')
run=json.loads(base.read_bytes())
source_graph=run['candidate']['sourceExecution']
for value in source_graph['kernels'].values():
    key=value['kernel']+'#'+value.get('entry','main')
    if key in digest_changes:
        assert value['digest']=='sha256:'+digest_changes[key]['before']
        value['digest']='sha256:'+digest_changes[key]['after']
graph_sha=sha(json.dumps(source_graph,sort_keys=True,separators=(',',':')).encode())
run['runtimeConfig']['inference']['kernelPath']['id']+='-stable-softplus-'+graph_sha[:12]
run['runtimeConfig']['inference']['kernelPath']['description']='Explicit stable-softplus candidate; unchanged frozen F32-KV source, weights, and acceptance policy.'
run['candidate']={'kind':'stable-softplus-only','parentRunSha256':sha(base.read_bytes()),
    'sourceExecution':source_graph,'canonicalSourceExecutionSha256':graph_sha,'shaderChanges':changes,'digestChanges':digest_changes}
run['runtimeRevision']='frozen-f32kv-20261107+stable-softplus-'+graph_sha[:12]
run['sourceFiles']=sorted(set(run['sourceFiles']+[row['file'] for row in changes]))
run['outputPath']=str(root/'observation.json')
run['captureRouterLayers']=list(range(48));run['captureLogitRows']=True
run['captureQsaLayers']=[];run['captureLayers']=False;run['lifecycle']=False
run['sequences']=[{'id':'long','maxTokens':1}]
(root/'run.json').write_text(json.dumps(run,indent=2)+'\n')
receipt={'scope':'Isolated ordinary-F32 runtime candidate: replace scalar softplus only; no reference tensor injection',
    'normalModelQualified':False,'producerSha256':sha(Path(__file__).read_bytes()),
    'replacementSha256':sha(replacement_source.read_bytes()),
    'parentRunSha256':sha(base.read_bytes()),'runSha256':sha((root/'run.json').read_bytes()),
    'canonicalSourceExecutionSha256':graph_sha,'changes':changes,'digestChanges':digest_changes,
    'artifactAndReferenceUnchanged':True,'candidateStatePrecision':'f32','serviceChanges':False}
(root/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'prepared':True,'shaders':len(changes),'digestChanges':len(digest_changes),'graph':graph_sha}))
