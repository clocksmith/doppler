import hashlib,json,pathlib,sys,tarfile
archive,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
rows=[]
with tarfile.open(archive) as source:
 for member in source.getmembers():
  assert member.name.startswith('package/')
  member.name=member.name[len('package/'):]
  source.extract(member,out,filter='data')
  if member.isfile():rows.append({'path':member.name,'bytes':member.size,'sha256':hashlib.sha256((out/member.name).read_bytes()).hexdigest()})
(out/'archive-files.json').write_text(json.dumps(rows)+'\n')
(out/'preparation.json').write_text(json.dumps({'archiveSha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'runtime':'immutable npm archive only'})+'\n')
