import collections,hashlib,json,pathlib,statistics,tarfile
root=pathlib.Path(__file__).resolve().parent
with tarfile.open(root.parent/'qwen-sampling-correction-20261017/doppler-gpu-0.6.32.tgz') as archive:
 inventory={v.name.removeprefix('package/'):hashlib.sha256(archive.extractfile(v).read()).hexdigest() for v in archive.getmembers() if v.isfile()}
def pct(values,p=.95):
 values=sorted(values);i=(len(values)-1)*p;lo=int(i);return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(i-lo)
def summarize(row):
 chunks=row['chunks'];first=chunks[0]['yielded'];events=row['events'];groups=[v for v in events if v['type']=='decode-group']
 return dict(firstTokenMs=first-row['start'],totalMs=row['end']-row['start'],decodeTokensPerSec=(len(row['tokens'])-1)*1000/(row['end']-first),deliveryP95Ms=pct([b['yielded']-a['yielded'] for a,b in zip(chunks,chunks[1:])]),streamQueueP95Ms=pct([v['dequeued']-v['yielded'] for v in chunks]),streamQueueMaxMs=max(v['dequeued']-v['yielded'] for v in chunks),domAppendP95Ms=pct([v['domUpdated']-v['dequeued'] for v in chunks]),rafOpportunityP95Ms=pct([v['callback']-chunks[v['index']]['domUpdated'] for v in row['frames']]),eventCounts=dict(collections.Counter(v['type'] for v in events)),groupTokens=[v['tokensRequested'] for v in groups],groupDurationMs=[v['end']-v['start'] for v in groups],queueSubmitCallMs=sum(v['end']-v['start'] for v in events if v['type']=='queue-submit'),javascriptSamplingMs=sum(v['end']-v['start'] for v in events if v['type']=='javascript-sampling'),stats={k:v for k,v in row['stats'].items() if k.startswith('decode') and isinstance(v,(int,float))})
result={'runtimeArchiveSha256':'75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2','model':'qwen-3-5-0-8b-q4k-ehaf16','workload':'64 raw input tokens; 128 greedy output tokens; 2 warmups then 3 untraced and 3 traced requests per invocation','candidateDelta':{'inference.session.decodeLoop.readbackInterval':{'baseline':4,'candidate':1}},'candidateDisposition':'rejected: throughput regression on both GPUs; no runtime/profile change retained','lanes':{}}
for lane in ['metal','vulkan']:
 runs={}
 for label,suffix in [('baselineBefore',''),('candidate','-interval1'),('baselineAfter','-queue-observed')]:
  path=root/f'{lane}-0-8b{suffix}.json';d=json.loads(path.read_text());assert d['passed']
  assert all(inventory.get(p.lstrip('/'))==h for p,h in d['files'].items())
  driver=next((p for p in root.glob('*.mjs') if hashlib.sha256(p.read_bytes()).hexdigest()==d['driverSha256']),None);assert driver,d['driverSha256']
  runs[label]={'receipt':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'driver':driver.name,'driverSha256':d['driverSha256'],'servedArchiveFilesVerified':len(d['files']),'untraced':[summarize(v) for v in d['result']['baseline']],'traced':[summarize(v) for v in d['result']['traced']]}
  if label=='baselineAfter':assert all(v['eventCounts'].get('queue-submit',0)>0 for v in runs[label]['traced'])
 result['lanes'][lane]=runs
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for lane,runs in result['lanes'].items():
 print(lane)
 for label,r in runs.items():
  print(label,{k:round(statistics.median(v[k] for v in r['untraced']),2) for k in ['decodeTokensPerSec','deliveryP95Ms','firstTokenMs','totalMs']})
 print('trace medians',{k:round(statistics.median(v[k] for v in runs['baselineAfter']['traced']),2) for k in ['queueSubmitCallMs','javascriptSamplingMs','streamQueueMaxMs','rafOpportunityP95Ms']})
