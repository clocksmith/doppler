"""Create a separate immutable numerical delivery; never mutate its parent."""
from pathlib import Path
import json,hashlib,shutil,tarfile
old=Path('/var/tmp/reploid-cold-compile-three-stage-20261013')
new=Path('/var/tmp/reploid-completion-candidate-20261013')
archive=Path('/var/tmp/doppler-completion-candidate-20261013/doppler-gpu-0.6.28.tgz')
sha=lambda b:hashlib.sha256(b).hexdigest()
parent_bytes=(old/'snapshot-binding.json').read_bytes()
assert sha(parent_bytes)=='804b41566f8e16e83425736c217f4416aae4c86bad9c56ae315194ff2b7bbf5d'
parent=json.loads(parent_bytes)
assert not new.exists(), 'Never overwrite an existing candidate delivery'
new.mkdir()
for row in parent['files']:
    path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
    data=(old/path).read_bytes();assert sha(data)==row['sha256'],str(path)
    (new/path).parent.mkdir(parents=True,exist_ok=True);(new/path).write_bytes(data)
raw=archive.read_bytes();assert sha(raw)=='2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8'
(new/'archive.tgz').write_bytes(raw)
with tarfile.open(archive) as t:
    for member in t.getmembers():
        path=Path(member.name);assert path.parts[0]=='package' and '..' not in path.parts
        if member.isdir(): continue
        assert member.isfile()
        target=new/'vendor/doppler/0.6.28'/Path(*path.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(t.extractfile(member).read())
paired=json.loads((new/'paired-binding.json').read_text());paired['archiveSha256']=sha(raw)
(new/'paired-binding.json').write_text(json.dumps(paired,indent=2)+'\n')
binding={**parent,'dopplerCommit':'a5f3ca4c','parentSnapshotSha256':sha(parent_bytes),'archiveSha256':sha(raw),
         'files':[{'path':r['path'],'sha256':sha((new/r['path']).read_bytes())} for r in parent['files']]}
(new/'snapshot-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
config=json.loads((old/'config-preparation.json').read_text().replace(str(old),str(new)))
for field in ['archive','pairedBinding','delivery']:
    config[field]['hash']='sha256:'+sha(Path(config[field]['path']).read_bytes())
(new/'config-preparation.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps({'root':str(new),'archiveSha256':sha(raw),'bindingSha256':sha((new/'snapshot-binding.json').read_bytes()),'files':len(binding['files'])}))
