import collections,hashlib,json,pathlib,statistics
root=pathlib.Path(__file__).resolve().parent
contract=json.loads((root/'campaign-contract-v3.json').read_text())
summary={'schema':'doppler.weight-representation-comparison/v1','baselineSha256':contract['baselineSha256'],
 'candidateSha256':contract['candidateSha256'],'unavailable':contract['unavailable'],'cases':[], 'comparisons':[],
 'limits':['Fresh browser/model opening; OS and driver caches not purged; local provisioned weights only.',
 'Two openings per engine/model/host, each followed by three short warm requests; no outliers removed.',
 'RSS sums browser process mappings every 500 ms; shared mappings can be counted repeatedly and short peaks missed.',
 'GPU counters cover the Doppler pool, not total vendor VRAM. ownedWeightBytes in raw receipts sums the loader tracking set and can include retired buffers; it is not used as resident memory.',
 'No separate CPU-time measurement. tensorLocations phase includes planning plus location construction wall time.',
 'Device recovery deliberately destroys the device after submission, not arbitrary spontaneous device loss.',
 'Token and full prefill-logit equality are scoped workload evidence, not general model correctness.']}
receipts={}
for host,lane in contract['hosts'].items():
 for case in lane['cases']:
  path=root/host/contract.get('outputDirectory','final')/case['id']/'receipt.json';record={'host':host,**case,'receipt':str(path.relative_to(root))}
  if not path.exists():record['status']='missing';summary['cases'].append(record);continue
  data=json.loads(path.read_text());record['receiptSha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  record['archiveSha256']=data['archiveSha256'];assert record['archiveSha256']==contract[case['engine']+'Sha256']
  record['status']='passed' if data['passed'] else 'failed';record['failure']=data.get('failure')
  if data['passed'] and case.get('driver')!='browser-materialization.mjs':
   rows=data['result']['rows'];opened=next(r for r in rows if r['stage']=='open');warm=[r for r in rows if r['stage']=='warm'];long=next(r for r in rows if r['stage']=='long-prefill')
   first=next(r for r in rows if r['stage']=='first');recreated=next(r for r in rows if r['stage']=='recreated')
   record.update({'modelManifestSha256':data['modelManifestSha256'],'configSha256':data['configSha256'],
    'browser':data['browser'],'openingMs':opened['openingMs'],'recreatedOpeningMs':recreated['openingMs'],
    'firstRequestTtftMs':first['firstTokenMs'],'firstRequestCompleteMs':first['completeMs'],
    'warmTtftMs':statistics.median(r['firstTokenMs'] for r in warm),
    'warmCompleteMs':statistics.median(r['completeMs'] for r in warm),
    'warmPrefillMs':statistics.median(r['stats']['prefillTimeMs'] for r in warm),
    'warmDecodeTokensPerSecond':statistics.median(r['stats']['decodeTokens']*1000/r['stats']['decodeTimeMs'] for r in warm),
    'longPrefillTokens':long['stats']['prefillTokens'],'longPrefillMs':long['stats']['prefillTimeMs'],
    'longTtftMs':long['firstTokenMs'],'longCompleteMs':long['completeMs'],
    'peakRssSumBytes':data['peakRssSumBytes'],
    'openingPoolAllocatedBytes':opened['memory']['pool']['currentBytesAllocated'],
    'openingPoolModelBytes':opened['memory']['pool']['resources']['retainedModel']['bytes'],
    'peakPoolAllocatedBytes':max(r.get('memory',{}).get('pool',r.get('pool',{})).get('peakBytesAllocated',0) for r in rows),
    'sourceBytesRead':opened['memory']['loader']['loadTiming']['bytesLoaded'],
    'tensorLocationsMs':opened['memory']['loader']['loadTiming']['phasesMs']['tensorLocations'],
    'planModes':dict(collections.Counter(v['mode'] for v in (opened['memory']['plan'] or {}).values()))})
   receipts[(host,case['id'])]=data
  elif data['passed']:record['fixture']=data['result']
  summary['cases'].append(record)
for host,lane in contract['hosts'].items():
 for size in ['2b','27b']:
  if not any(c['id'].startswith('baseline-'+size) for c in lane['cases']):continue
  ids=[f'{engine}-{size}-{order}' for engine,order in [('baseline','first'),('candidate','first'),('candidate','second'),('baseline','second')]]
  data=[receipts.get((host,c)) for c in ids];comparison={'host':host,'model':size,'order':ids,'complete':all(data)}
  if all(data):
   comparison['sameManifest']=len({d['modelManifestSha256'] for d in data})==1
   comparison['sameConfig']=len({d['configSha256'] for d in data})==1
   logits=[next(r['logitsSha256'] for r in d['result']['rows'] if r['stage']=='numerical') for d in data]
   comparison['allPrefillLogitsByteIdentical']=len(set(logits))==1
   short=[r for d in data for r in d['result']['rows'] if r['stage'] in ['first','warm']]
   long=[r for d in data for r in d['result']['rows'] if r['stage']=='long-prefill']
   comparison['allShortTokensIdentical']=all(r['tokens']==short[0]['tokens'] and r['text']==short[0]['text'] for r in short)
   comparison['allLongTokensIdentical']=all(r['tokens']==long[0]['tokens'] and r['text']==long[0]['text'] for r in long)
   comparison['allLifecyclePassed']=all(d['passed'] for d in data)
   unloaded=[next(r['pool']['resources'] for r in d['result']['rows'] if r['stage']=='unloaded') for d in data]
   comparison['allUnloadOwnershipReleased']=all(all(r[k]['bytes']==0 and r[k]['count']==0 for k in ['retainedModel','active','deferredCleanup']) for r in unloaded)
   comparison['gatePassed']=all(comparison[k] for k in ['sameManifest','sameConfig','allPrefillLogitsByteIdentical','allShortTokensIdentical','allLongTokensIdentical','allLifecyclePassed','allUnloadOwnershipReleased'])
  summary['comparisons'].append(comparison)
integrity=[]
for host,size in [('metal','2b'),('vulkan','2b'),('vulkan','27b')]:
 path=root/host/f'model-{size}-piece-integrity.json'
 record={'host':host,'model':size,'receipt':str(path.relative_to(root)),'passed':False}
 if path.exists():
  data=json.loads(path.read_text())
  manifests={c['modelManifestSha256'] for c in summary['cases'] if c['host']==host and c['id'].startswith(('baseline-'+size+'-','candidate-'+size+'-')) and 'modelManifestSha256' in c}
  record.update({'passed':data['passed'] and manifests=={data['manifestSha256']},'receiptSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(data['files']),'bytes':sum(f['bytes'] for f in data['files'])})
 integrity.append(record)
summary['modelIntegrity']=integrity
paths=[root/host/'model-2b-piece-integrity.json' for host in ['metal','vulkan']]
summary['same2bBytesAcrossHosts']=all(p.exists() for p in paths) and len({json.dumps(sorted((f['file'],f['bytes'],f['sha256']) for f in json.loads(p.read_text())['files'])) for p in paths})==1
summary['modelIntegrityPassed']=all(r['passed'] for r in integrity) and summary['same2bBytesAcrossHosts']
summary['availableMatrixComplete']=all(c['status']=='passed' for c in summary['cases']) and all(c.get('gatePassed',False) for c in summary['comparisons'])
summary['availableMatrixComplete']=summary['availableMatrixComplete'] and summary['modelIntegrityPassed']
summary['fullRequestedMatrixComplete']=summary['availableMatrixComplete'] and not summary['unavailable']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Frozen baseline versus candidate','', '| Host / model / engine / order | Open ms | Warm TTFT ms | Warm decode tok/s | Pool model MiB | Peak pool MiB | Peak summed RSS MiB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in summary['cases']:
 if 'openingMs' not in r:continue
 label=f"{r['host']} / {r['id']}"
 lines.append(f"| {label} | {r['openingMs']:.1f} | {r['warmTtftMs']:.1f} | {r['warmDecodeTokensPerSecond']:.2f} | {r['openingPoolModelBytes']/2**20:.1f} | {r['peakPoolAllocatedBytes']/2**20:.1f} | {r['peakRssSumBytes']/2**20:.1f} |")
lines+=['','See `summary.json` for long prefill, first request, recreated-device opening, failure and identity details. These are scoped measurements, not a general speedup claim.','']
(root/'comparison-table.md').write_text('\n'.join(lines))
print(json.dumps({'availableMatrixComplete':summary['availableMatrixComplete'],'fullRequestedMatrixComplete':summary['fullRequestedMatrixComplete'],'comparisons':summary['comparisons']}))
