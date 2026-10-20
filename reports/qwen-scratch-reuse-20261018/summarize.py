"""Reduce fixed comparisons and retain unfavorable outcomes."""
import hashlib,json,pathlib,random,statistics
p=pathlib.Path(__file__).resolve().parent
read=lambda name:json.loads((p/name).read_text())
contract=read('contract.json');baseline=read('baseline-reproduction.json');candidate=read('candidate-lifecycle.json')
assert baseline['passed'] and candidate['passed']
expected=next(r for r in baseline['result']['rows']if r['stage']=='first')
long=next(r for r in baseline['result']['rows']if r['stage']=='long-prefill')
logits=next(r for r in baseline['result']['rows']if r['stage']=='numerical')['logitsSha256']
archives={'baseline':contract['baselineArchiveSha256'],'candidate':'addf3788637a68ed929c20445d7a540e6a5bd417811284e8a5a11737b6c4284d'}
def numerical(receipt):
 assert receipt['passed']
 assert receipt['modelManifestSha256']==contract['modelManifestSha256']
 for row in receipt['result']['rows']:
  if 'tokens' in row:
   control=long if row['stage']=='long-prefill' else expected
   assert row['tokens']==control['tokens'] and row['text']==control['text']
  if row['stage']=='numerical':assert row['logitsSha256']==logits
numerical(candidate)
metrics={arm:[]for arm in archives}
for i in range(contract['primary']['pairs']):
 for arm in archives:
  r=read(f'pair-{i:02d}-{arm}.json');numerical(r)
  assert r['archiveSha256']==archives[arm] and r['configSha256']==baseline['configSha256']
  assert r['browser']==baseline['browser']
  assert r['hardware']['gpu']['devices']==baseline['hardware']['gpu']['devices']
  assert r['driverSha256']==hashlib.sha256((p/'run.js').read_bytes()).hexdigest()
  rows=r['result']['rows'];warm=[x for x in rows if x['stage']=='warm'];assert len(warm)==5
  prev=next(x for x in rows if x['stage']=='first')['pool'];alloc=[];reuse=[]
  for x in warm:
   alloc.append(x['pool']['allocations']-prev['allocations']);reuse.append(x['pool']['reuses']-prev['reuses']);prev=x['pool']
  m=lambda fn:statistics.median(fn(x)for x in warm)
  metrics[arm].append({'openingMs':rows[0]['openingMs'],'firstOutputMs':m(lambda x:x['firstTokenMs']),
   'completeMs':m(lambda x:x['completeMs']),'decodeMs':m(lambda x:x['stats']['decodeTimeMs']),
   'decodeTokensPerSecond':m(lambda x:1000*x['stats']['decodeTokens']/x['stats']['decodeTimeMs']),
   'cpuRecordingMs':m(lambda x:x['stats']['decodeRecordMs']),
   'prefillMs':m(lambda x:x['stats']['prefillTimeMs']),
   'submitWaitMs':m(lambda x:x['stats']['decodeSubmitWaitMs']),
   'readbackWaitMs':m(lambda x:x['stats']['decodeReadbackWaitMs']),
   'readbackMapWaitMs':m(lambda x:x['stats']['decodeReadbackMapWaitMs']),
   'readbackCleanupMs':m(lambda x:x['stats']['decodeReadbackCleanupMs']),
   'readbackCopyMs':m(lambda x:x['stats']['decodeReadbackCopyMs']),
   'recordedDispatches':m(lambda x:x['stats']['decodeRecordOps']),
   'recordedPasses':m(lambda x:x['stats']['decodeRecordPasses']),
   'newPoolAllocations':statistics.median(alloc),'poolReuses':statistics.median(reuse),
   'peakPoolBytes':max(x['pool']['peakBytesAllocated']for x in warm),'peakRssSumBytes':r['peakRssSumBytes']})
gains=[100*(1-c['completeMs']/b['completeMs'])for b,c in zip(metrics['baseline'],metrics['candidate'])]
rng=random.Random(0);boot=sorted(statistics.median(rng.choices(gains,k=len(gains)))for _ in range(20000));ci=[boot[500],boot[19499]]
medians={arm:{key:statistics.median(r[key]for r in rows)for key in rows[0]}for arm,rows in metrics.items()}
regressions={key:100*(medians['candidate'][key]/medians['baseline'][key]-1)for key in ['firstOutputMs','decodeMs','peakPoolBytes','peakRssSumBytes']}
accepted=statistics.median(gains)>=2 and ci[0]>0 and all(x<=3 for x in regressions.values())
probes={}
for arm in archives:
 r=read(f'{arm}-probe.json');numerical(r);row=next(x for x in r['result']['rows']if x['stage']=='memory')
 a=row['attribution'];zero={'calls':0,'ms':0}
 totals=lambda table:{'calls':sum(x['calls']for x in table.values()),'ms':sum(x['ms']for x in table.values())}
 probes[arm]={'allAcquire':totals(a['acquire']),'allCreate':totals(a['create']),
  'ffnInputAcquire':a['acquire'].get('fused_ffn_output',zero),
  'ffnInputCreate':totals({k:v for k,v in a['create'].items()if k.startswith('fused_ffn_output_')}),
  'fusedOutputs':a['fusedOutputs'],'scratchBuffers':a['scratchBuffers'],
  'pool':row['memory']['pool'],'ownedWeightBytes':row['memory']['ownedWeightBytes']}
summary={'schema':'doppler.scratch-reuse-result/v1','accepted':accepted,'disposition':'retain'if accepted else'reject: complete-request/memory gates not established',
 'contractSha256':hashlib.sha256((p/'contract.json').read_bytes()).hexdigest(),'archives':archives,'modelManifestSha256':contract['modelManifestSha256'],
 'pairs':metrics,'medians':medians,'pairedCompleteGainPercent':gains,'medianPairedGainPercent':statistics.median(gains),'pairedBootstrap95Percent':ci,'regressionPercent':regressions,
 'numerical':'Exact short/long tokens, text and public-prefill-logit byte digest on M3; not general correctness or task quality',
 'vulkanSuccessorQualification':read('vulkan-qualification.json'),
 'lifecycle':{'m3CandidatePassed':True,'stages':[x['stage']for x in candidate['result']['rows']]},'attribution':probes,
 'limits':['Eight opening pairs, five warm requests each; opening-level medians bootstrap, not 40 independent observations','GPU allocated pool bytes exclude driver allocations and raw buffers; unchanged session slot is preallocated in both arms','RSS sums may double-count shared pages; no claim of exclusive host','Submit and readback waits overlap; do not add them']}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k]for k in ['accepted','medians','pairedCompleteGainPercent','medianPairedGainPercent','pairedBootstrap95Percent','regressionPercent']},indent=2))
