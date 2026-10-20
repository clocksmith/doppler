"""Build an isolated comparison checkout around immutable npm runtime bytes."""
import hashlib,io,json,pathlib,subprocess,sys,tarfile
repo,archive=map(pathlib.Path,sys.argv[1:3]);commit=sys.argv[3];out=pathlib.Path(sys.argv[4])
repo=repo.resolve();archive=archive.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
paths=['tools','tests','benchmarks','assets','demo','package.json','package-lock.json','models/catalog.json']
source=subprocess.check_output(['git','archive',commit,*paths],cwd=repo)
with tarfile.open(fileobj=io.BytesIO(source)) as t:t.extractall(out,filter='data')
records=[]
with tarfile.open(archive) as t:
 for member in t.getmembers():
  if not member.name.startswith('package/'):raise ValueError(member.name)
  member.name=member.name[len('package/'):]
  t.extract(member,out,filter='data')
  if member.isfile():records.append(dict(path=member.name,sha256=hashlib.sha256((out/member.name).read_bytes()).hexdigest(),bytes=member.size))
support=[]
for name in ['src/tooling/hf-registry/registry-urls.js','src/tooling/hf-registry/catalog-io.js','src/formats/plain-object.js']:
 content=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo)
 dest=out/'.harness-support'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
 support.append(dict(path=str(dest.relative_to(out)),sha256=hashlib.sha256(content).hexdigest()))
comparator=out/'tools/compare-engines.js';text=comparator.read_text();needle="'../src/tooling/hf-registry/registry-urls.js'"
assert text.count(needle)==1,'Comparator must import the canonical registry URL owner'
comparator.write_text(text.replace(needle,"'../.harness-support/src/tooling/hf-registry/registry-urls.js'"))
(out/'node_modules').symlink_to(repo/'node_modules',target_is_directory=True)
(out/'archive-files.json').write_text(json.dumps(records,indent=2)+'\n')
receipt=dict(archiveSha256=hashlib.sha256(archive.read_bytes()).hexdigest(),driverCommit=commit,
 runtime='npm archive only; no checkout src fallback',harnessSupport=support,
 comparatorImportRewrite='registry URL helper only; outside runtime src',
 comparatorSha256=hashlib.sha256(comparator.read_bytes()).hexdigest(),dependencies=str(repo/'node_modules'))
(out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['node','tools/compare-engines.js','--help'],cwd=out,check=True,stdout=subprocess.DEVNULL)
print(json.dumps(receipt,indent=2))
