from pathlib import Path
import json,sys,math,statistics,random,hashlib
root=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent)
contract=json.loads((root/'product-repeat-contract.json').read_text())
receipt=json.loads((root/'product-repeat/paired-receipt.json').read_text())
assert receipt.get('complete'), 'Incomplete repeated comparison'
rows=[];groups={};eos={248044,248046}
for prompt in contract['shared']['prompts']:
 p=prompt['id'];group=[]
 teacher=json.loads((root/'full-logits-baseline'/f'gate-{p}-baseline.json').read_text())['result']['metrics']['referenceTranscript']
 for i in range(contract['shared']['pairsPerPrompt']):
  dp=root/'product-repeat'/f'{p}-pair{i}-doppler.json';pp=root/'product-repeat'/f'{p}-pair{i}-pooled.json'
  d=json.loads(dp.read_text())['result'];v=json.loads(pp.read_text());pool=next(x for x in v['results'] if not x['warmup'])
  m=d['metrics'];run=m['requestTiming']['samples'][0];dt=m['referenceTranscript'];stop=bool(eos.intersection(pool['tokenIds']) or eos.intersection(dt['tokens']['ids']))
  checks={'promptIdsMatch':dt['prompt']['ids']==pool['promptIds']==teacher['prompt']['ids'],'dopplerTeacherMatch':dt['tokens']['ids']==teacher['tokens']['ids'],'generated64Each':len(dt['tokens']['ids'])==len(pool['tokenIds'])==64,'noForcedContinuationPastEOS':not stop,'browserMatches':v['userAgent']==d['env']['browserUserAgent'],'physicalAMD':d['deviceInfo']['adapterInfo']['vendor'].lower()=='amd' and v['adapterInfo']['vendor'].lower()=='amd','noBrowserErrors':not v.get('errors') and not v.get('gpuErrors'),'requestTimingScope':run['scope']=='generation-call-through-completion'}
  assert all(checks.values()), (p,i,checks)
  row={'prompt':p,'pair':i,'doppler':{k:run[k] for k in ['totalRunMs','firstTokenMs','decodeMs','decodeTokensPerSec']},'pooled':{k:pool[k] for k in ['totalRunMs','firstTokenMs','decodeMs','decodeTokensPerSec']},'checks':checks,'sha256':{'doppler':hashlib.sha256(dp.read_bytes()).hexdigest(),'pooled':hashlib.sha256(pp.read_bytes()).hexdigest()}}
  row['completeRatio']=run['totalRunMs']/pool['totalRunMs'];row['logRatio']=math.log(row['completeRatio']);rows.append(row);group.append(row)
 groups[p]=group
mean=lambda a:statistics.mean(a)
ratio=math.exp(mean(mean(x['logRatio'] for x in g) for g in groups.values()))
rng=random.Random(0);samples=sorted(math.exp(mean(mean(rng.choice(g)['logRatio'] for _ in g) for g in groups.values())) for _ in range(10000))
interval=[samples[250],samples[9749]]
summary=[]
for p,g in groups.items():
 q={'prompt':p,'pairs':len(g),'median':{engine:{key:statistics.median(x[engine][key] for x in g) for key in ['totalRunMs','firstTokenMs','decodeMs','decodeTokensPerSec']} for engine in ['doppler','pooled']}}
 q['completeMedianRatio']=q['median']['doppler']['totalRunMs']/q['median']['pooled']['totalRunMs'];q['ttftMedianRatio']=q['median']['doppler']['firstTokenMs']/q['median']['pooled']['firstTokenMs'];summary.append(q)
gates={'minimum2PercentGain':ratio<=0.98,'upper95RatioBelow1':interval[1]<1,'perPromptMedianCompleteRegressionAtMost3Percent':all(x['completeMedianRatio']<=1.03 for x in summary),'perPromptMedianTTFTRegressionAtMost5Percent':all(x['ttftMedianRatio']<=1.05 for x in summary),'allComparableWorkChecks':True}
out={'schema':'doppler.jupiter-product-comparison-summary.v1','scope':contract['scope'],'pairs':len(rows),'attempts':len(receipt['sessions']),'outliersRemoved':0,'completeRequestRatio':ratio,'completeRequestGainPercent':100*(1-ratio),'ratio95Interval':interval,'gates':gates,'targetMet':all(gates.values()),'prompts':summary,'rows':rows,'limitations':['Different reconstructed weights and generated outputs; product-format measurements only, not equal-weight engine attribution or application quality.','Doppler returns its remaining 63 tokens in one burst in this 64-token workload; Pooled delivers one at a time. Delivery latency is retained in raw samples.','Warm resident inference after one same-prompt warmup per fresh browser; load and compile excluded from request timing.','Existing desktop Chrome GPU process remained open; no other tuning process was intentionally run concurrently.','One model, one AMD GPU, one browser; no broader default or artifact promotion.']}
(root/'product-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print('ratio',ratio,'95%',interval,'targetMet',out['targetMet'])
for p in summary:print(p['prompt'],p['median'], 'ratios',p['completeMedianRatio'],p['ttftMedianRatio'])
