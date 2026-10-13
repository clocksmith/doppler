"""Isolated candidate extraction install; not signed application qualification."""
from pathlib import Path
import json,hashlib,base64,shutil,tarfile
old=Path('/var/tmp/reploid-recurrent-memory-20261013/installed-cold/reploid')
root=Path('/var/tmp/reploid-completion-candidate-20261013');app=root/'installed/reploid'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not app.exists(), 'Never overwrite an installed candidate'
original=(old/'native-delivery.json').read_bytes()
assert sha(original)=='38e4fa9e29cb5d9aec713da8ecee270bdb082e6382e4117364e8e0ff140805a6'
delivery=json.loads(original)
for row in delivery['entries']: assert sha((old/row['path']).read_bytes())==row['sha256']
archive=root/'archive.tgz';raw=archive.read_bytes()
assert sha(raw)=='2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8'
integrity='sha512-'+base64.b64encode(hashlib.sha512(raw).digest()).decode()
shutil.copytree(old,app,symlinks=False)
(app/'deploy/artifacts/doppler-gpu-0.6.28.tgz').write_bytes(raw)
pin_path=app/'self/config/doppler-package.json';pin=json.loads(pin_path.read_text())
pin['integrity']=integrity
if 'sha256' in pin:pin['sha256']=sha(raw)
pin_path.write_text(json.dumps(pin,indent=2)+'\n')
lock_path=app/'package-lock.json';lock=json.loads(lock_path.read_text());lock['packages']['node_modules/doppler-gpu']['integrity']=integrity
lock_path.write_text(json.dumps(lock,indent=2)+'\n')
# Archive dependency metadata is unchanged; extraction replaces exactly its members.
runtime=app/'node_modules/doppler-gpu';shutil.rmtree(runtime);runtime.mkdir()
with tarfile.open(archive) as t:
    for m in t.getmembers():
        p=Path(m.name);assert p.parts[0]=='package' and '..' not in p.parts
        if m.isdir():continue
        assert m.isfile();dest=runtime/Path(*p.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(t.extractfile(m).read())
delivery['runtime']['integrity']=integrity
for row in delivery['entries']:
    data=(app/row['path']).read_bytes();row['sha256']=sha(data)
    if 'bytes' in row:row['bytes']=len(data)
    if 'sizeBytes' in row:row['sizeBytes']=len(data)
(app/'native-delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
config={'applicationDirectory':str(app),'nativeDeliverySha256':sha((app/'native-delivery.json').read_bytes()),
        'providerModule':'webgpu','testDelivery':{'root':str(root),'bindingPath':str(root/'snapshot-binding.json'),
        'bindingSha256':sha((root/'snapshot-binding.json').read_bytes()),'archivePath':str(archive)}}
(root/'startup-config.json').write_text(json.dumps(config,indent=2)+'\n')
receipt={'schema':'doppler.candidate-extraction-install/v1','method':'Verified archive extraction; original application/native provider files retained',
         'parentNativeDeliverySha256':sha(original),'runtimeArchiveSha256':sha(raw),'startupConfig':config,
         'signedApplicationQualified':False,'publicPackageChanged':False}
(root/'installed-candidate.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
