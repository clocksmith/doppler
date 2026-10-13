"""Compare installed candidate output to the unchanged retained F64 reference."""
from pathlib import Path
import json,gzip,base64,hashlib,struct,math
from datetime import datetime
ROOT=Path(__file__).resolve().parent
path=ROOT/'installed-prefix-result.json'
raw=path.read_bytes() if path.exists() else gzip.decompress((ROOT/'installed-prefix-result.json.gz').read_bytes())
result=json.loads(raw)
reference_raw=gzip.decompress((ROOT.parent/'ordered-partitions-20261013/long-prefill-boundary/observation.json.gz').read_bytes())
assert hashlib.sha256(reference_raw).hexdigest()=='260469b51eb89d1532a9f58f98d43a0b26a821b964559bb529b39c991ec61744'
reference=json.loads(reference_raw)
assert result['delivery']['archiveSha256']=='2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8'
assert len(result['cases'])==1
observed=result['cases'][0]
a=[v[0] for v in struct.iter_unpack('<f',base64.b64decode(observed['logits']))]
b=[v[0] for v in struct.iter_unpack('<d',base64.b64decode(reference['referenceLogits']))]
assert len(a)==len(b)==248320
errors=[abs(x-y) if math.isfinite(x) else math.inf for x,y in zip(a,b)]
expected=max(range(len(b)),key=b.__getitem__)
operations=[]
for op in result['operations']['operations']:
    operations.append({k:op[k] for k in ['label','status','startedAt','finishedAt']})
    operations[-1]['elapsedMs']=(datetime.fromisoformat(op['finishedAt'])-datetime.fromisoformat(op['startedAt'])).total_seconds()*1000
close=result.get('providerSettlement',{})
failures={k:result[k] for k in ['failure','cleanupFailure','providerError','contextError','observerCloseError'] if result.get(k)}
peak=result['settlement']['device']['peakBytes']
report={'schema':'doppler.installed-prefix-comparison/v1','scope':'One original output-stage prefix on Intel/native; not complete55, all placements or application qualification',
        'archiveSha256':result['delivery']['archiveSha256'],'installedFiles':result['delivery']['installedFiles'],
        'resultSha256':hashlib.sha256(raw).hexdigest(),'referenceObservationSha256':hashlib.sha256(reference_raw).hexdigest(),
        'logits':len(a),'maxAbsoluteError':max(errors),'tolerance':.001,'failures':sum(e>.001 for e in errors),
        'actualToken':observed['tokenId'],'expectedToken':expected,'operations':operations,'lifecycleErrors':failures,
        'providerSettlement':close,'peakAllocatedGpuBytes':peak,'gpuLimitBytes':1420000000,
        'power':{'available':False,'reason':'RAPL energy_uj unreadable by diagnostic account'},
        'transfer':{'scope':'Loopback model server, not live peer acquisition','measured':False},
        'passed':max(errors)<=.001 and observed['tokenId']==expected and not failures and close.get('closed') is True and peak<=1420000000,
        'completeQualification':False,'promotion':False}
(ROOT/'installed-prefix-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['archiveSha256','maxAbsoluteError','actualToken','passed','completeQualification']}))
if not report['passed']:raise SystemExit(1)
