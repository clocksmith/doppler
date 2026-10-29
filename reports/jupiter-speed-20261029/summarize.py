"""Apply the frozen gates; retain all sessions and failures."""
from pathlib import Path
import array,base64,hashlib,json,math,random,statistics
ROOT=Path(__file__).resolve().parent
contract=json.loads((ROOT/'contract.json').read_text())
r=json.loads((ROOT/'runs/receipt.json').read_text());assert r.get('complete')
median=statistics.median
prompts=[];logs=[];audits=[]
for p in contract['prompts']:
 rows=[s for s in r['sessions'] if s['prompt']==p['id']]
 a=[s for s in rows if s['lane']=='baseline'];b=[s for s in rows if s['lane']=='candidate']
 assert len(a)==len(b)==contract['pairsPerPrompt']
 pairs=[(next(s for s in a if s['pair']==i),next(s for s in b if s['pair']==i)) for i in range(contract['pairsPerPrompt'])]
 log_ratios=[math.log(y['metrics']['totalRunMs']/x['metrics']['totalRunMs']) for x,y in pairs];logs.append(log_ratios)
 def stats(lane,key):
  vals=[s['metrics'][key] for s in lane];return {'samples':vals,'median':median(vals),'min':min(vals),'max':max(vals)}
 same=all(s['promptIds']==rows[0]['promptIds'] and s['outputIds']==rows[0]['outputIds'] and len(s['outputIds'])==64 for s in rows)
 record={'id':p['id'],'promptTokens':len(rows[0]['promptIds']),'generatedTokens':len(rows[0]['outputIds']),'sameInputsAndOutputs':same,'baseline':{k:stats(a,k) for k in ['totalRunMs','decodeTokensPerSec','firstTokenMs','modelLoadMs']},'candidate':{k:stats(b,k) for k in ['totalRunMs','decodeTokensPerSec','firstTokenMs','modelLoadMs']},'pairedCompleteGainPct':[(1-math.exp(v))*100 for v in log_ratios],'perPromptCompleteRatio':median(s['metrics']['totalRunMs'] for s in b)/median(s['metrics']['totalRunMs'] for s in a),'perPromptFirstTokenRatio':median(s['metrics']['firstTokenMs'] for s in b)/median(s['metrics']['firstTokenMs'] for s in a)}
 prompts.append(record)
 for s in [a[0],b[0]]:
  raw=json.loads((ROOT/'runs'/f"{s['id']}.json").read_text())['result'];m=raw['metrics'];resolved=raw['request']['runtimeConfig'];audits.append({'id':s['id'],'requested':json.loads((ROOT/'runs'/f"{s['id']}-config.json").read_text())['request'],'resolvedCommandSettings':resolved,'observedExecution':{k:m.get(k) for k in ['decodeMode','batchGuardReason','executionPlan','decodeCadence','executionContractArtifact']},'observedAllocationCounters':raw.get('memoryStats'),'greedyHeadFusionEligibility':{'requested':resolved['inference']['session']['useGreedyLmHeadArgmaxFusion'],'eligible':False,'blockingCondition':'temperature < greedyThreshold evaluates 0 < 0, false in generator-steps.js shouldUseGreedyLmHeadArgmaxFusion','temperature':0,'greedyThreshold':0,'observedFusedHeadDispatch':'Absent from untimed dispatch audit; ordinary sample_argmax/argmax_reduce present'},'limits':'Resolved command settings precede manifest compile; observed execution plan/cadence and untimed dispatch audit separately establish what executed. Allocation counters are not residency.'})
rng=random.Random(20261029);boot=sorted(math.exp(statistics.mean(statistics.mean(rng.choices(xs,k=len(xs))) for xs in logs)) for _ in range(10000))
ratio=math.exp(statistics.mean(statistics.mean(xs) for xs in logs));ci=[boot[249],boot[9749]]
performance={'estimator':'equal-weight per-prompt mean paired log complete-request latency ratio, exponentiated','candidateToBaselineRatio':ratio,'gainPct':100*(1-ratio),'bootstrap95RatioInterval':ci,'minimumGainPassed':ratio<=0.98,'positiveBoundPassed':ci[1]<1,'perPromptCompleteRegressionPassed':all(p['perPromptCompleteRatio']<=1.03 for p in prompts),'perPromptFirstTokenRegressionPassed':all(p['perPromptFirstTokenRatio']<=1.05 for p in prompts)}
performance['passed']=all(performance[k] for k in ['minimumGainPassed','positiveBoundPassed','perPromptCompleteRegressionPassed','perPromptFirstTokenRegressionPassed'])
gate_receipt=json.loads((ROOT/'gates/receipt.json').read_text());assert gate_receipt.get('complete')
logit=[];dispatch=[]
for p in contract['prompts']:
 raw={lane:json.loads((ROOT/'gates'/f"{p['id']}-{lane}.json").read_text())['result'] for lane in ['baseline','candidate']}
 heads={k:v['metrics']['fullLogitGate'] for k,v in raw.items()};assert heads['baseline']['teacher']==heads['candidate']['teacher']
 rows=[]
 for a,b in zip(heads['baseline']['rows'],heads['candidate']['rows']):
  assert a['position']==b['position'] and a['elements']==b['elements']==248320
  av=array.array('f');av.frombytes(base64.b64decode(a['base64']));bv=array.array('f');bv.frombytes(base64.b64decode(b['base64']))
  finite=all(math.isfinite(x) and math.isfinite(y) for x,y in zip(av,bv));errors=[abs(x-y) for x,y in zip(av,bv)];failures=sum(v>0.001 for v in errors)
  rows.append({'position':a['position'],'elements':len(av),'finite':finite,'maxAbs':max(errors),'failures':failures})
 assert [x['position'] for x in rows]==[0,1,2,4,8,16,32,63]
 token_equal=raw['baseline']['metrics']['referenceTranscript']['tokens']['ids']==raw['candidate']['metrics']['referenceTranscript']['tokens']['ids']==heads['baseline']['teacher']['outputIds']
 logit.append({'prompt':p['id'],'rows':rows,'sameGreedyTokens':token_equal,'passed':token_equal and all(x['finite'] and x['failures']==0 for x in rows)})
 for lane,v in raw.items():dispatch.append({'prompt':p['id'],'lane':lane,**v['metrics']['dispatchAudit']})
summary={'schema':'doppler.jupiter-speed-result.v1','scope':'Qwen3.5 0.8B/Radeon8060S/Chrome146; opt-in experimental configuration, no release or cross-engine claim','timedSessions':len(r['sessions']),'failedTimedAttempts':r['failures'],'failedGateAttempts':gate_receipt['failures'],'prompts':prompts,'performanceGate':performance,'fullLogitGate':logit,'atol':0.001,'rtol':0,'accepted':performance['passed'] and all(p['sameInputsAndOutputs'] for p in prompts) and all(g['passed'] for g in logit) and not r['failures'] and not gate_receipt['failures'],'limits':contract['limits']+['Four prompts, 64 generated tokens each, five pairs/prompt; no general quality, long-decode, cold-disk or tail guarantee.','Logit gate materializes the full head; fused argmax intermediate logits are not exposed.','Full-logit and dispatch audit runs are untimed and excluded from the speed population.','Shared host; no other service stopped. Initial source-identity capture ran during timing; every timing sample retained.']}
for name,d in [('summary.json',summary),('config-audit.json',audits),('dispatch-audit.json',dispatch)]: (ROOT/name).write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'accepted':summary['accepted'],'performance':performance,'logitPassed':[x['passed'] for x in logit]},indent=2))
