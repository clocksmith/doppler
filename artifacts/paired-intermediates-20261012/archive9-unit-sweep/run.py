from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
root=Path(sys.argv[1]);ev=root/'evidence';b=json.loads((ev/'binding.json').read_text());source=Path(b['snapshot']);env=dict(os.environ)
for k in list(env):
 if k.startswith('GIT_'):del env[k]
env.update(b['env']);now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();sha=lambda x:hashlib.sha256(x).hexdigest()
r={'schema':'doppler.frozen-unit-sweep/v1','startedAt':now(),'pid':os.getpid(),'command':b['command'],'cwd':str(source),'status':'running'}
(ev/'run.json').write_text(json.dumps(r,indent=2)+'\n')
with (ev/'unit.log').open('wb') as out:
 p=subprocess.Popen(b['command'],cwd=source,env=env,stdout=out,stderr=subprocess.STDOUT);r['childPid']=p.pid;(ev/'run.json').write_text(json.dumps(r,indent=2)+'\n');status=p.wait()
r.update(status='completed',exitCode=status,finishedAt=now());(ev/'run.json').write_text(json.dumps(r,indent=2)+'\n')
before=json.loads((ev/'tracked-before.json').read_text());diff=[]
for name,h in before.items():
 p=source/name
 if not p.is_file() or sha(p.read_bytes())!=h:diff.append(name)
after={'sourceCommit':b['sourceCommit'],'trackedFiles':len(before),'trackedByteDifferences':diff,'archiveSha256':sha(Path(b['archive']).read_bytes()),'installedDependencyLockSha256':sha((Path(b['dependencyDirectory'])/'.package-lock.json').read_bytes()),'gitHead':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True,env=env).strip(),'gitDiff':subprocess.check_output(['git','-C',str(source),'diff','--name-only',b['sourceCommit']],text=True,env=env).splitlines()}
(ev/'source-after.json').write_text(json.dumps(after,indent=2)+'\n')
assert not diff and not after['gitDiff'];assert after['archiveSha256']==b['archiveSha256'];assert after['installedDependencyLockSha256']==b['installedDependencyLockSha256'];assert after['gitHead']==b['sourceCommit']
print(json.dumps(r))
