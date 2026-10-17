import hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parent
lanes={'metal':root/'comparison-metal-selected','vulkan':root/'comparison-vulkan-preliminary'}
rows=[];attempts=[]
for lane,directory in lanes.items():
 for p in sorted(directory.glob('*/status.json')):
  attempts.append(dict(lane=lane,**json.loads(p.read_text())))
 for p in sorted(directory.glob('*/compare_latest.json')):
  d=json.loads(p.read_text());mode=d['mode'];s=d['sections'][mode];dr=s['doppler']['result'];tr=s['transformersjs']
  opening=None
  if mode=='cold':
   ds=dr.get('metrics',{}).get('requestTiming',{}).get('samples',[]);ts=tr.get('runs',[])
   if ds and ts:
    opening=dict(scope='model initialization plus first generation; excludes browser/process launch, network download and OS/driver cache purge',independentOpeningsPerReceipt=1,residentRequestsPerOpening=d['runs'],doppler=dict(loadMs=dr['timing']['modelLoadMs'],firstTokenMs=ds[0]['firstTokenMs'],firstResponseMs=dr['timing']['modelLoadMs']+ds[0]['firstTokenMs'],completeMs=dr['timing']['modelLoadMs']+ds[0]['totalRunMs']),transformersjs=dict(loadMs=tr['timing']['modelLoadMs'],firstTokenMs=ts[0]['ttftMs'],firstResponseMs=tr['timing']['modelLoadMs']+ts[0]['ttftMs'],completeMs=tr['timing']['modelLoadMs']+ts[0]['totalMs']))
  rows.append(dict(lane=lane,mode=mode,model=d['dopplerModelId'],promptTokens=d['workload']['prefillTokenTarget'],order=d['methodology'][('cold' if mode=='cold' else 'warm')+'EngineOrder'],receipt=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),archiveSha256='75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2',command=json.loads((p.parent/'command.json').read_text()),fairness=d['fairness'],outputParity=s['outputParity'],doppler=dr['timing'],transformersjs=tr['timing'],coldOpening=opening,memoryAccounting=s['memoryAccounting'],resourceTelemetry=s['resourceTelemetry'],dopplerBottleneck=s['dopplerBottleneck'],dopplerBatchAccounting=s['dopplerBatchAccounting'],device=dr['deviceInfo'],comparatorDevice=tr['deviceInfo'],comparatorStack=d['comparatorStack']))
successful=len(rows)==24 and len(attempts)==24 and all(a.get('exitCode')==0 for a in attempts) and all(r['fairness']['claimGrade'] and r['outputParity']['status']=='match' for r in rows)
summary=dict(complete=successful,rows=rows,attempts=attempts,limits=['No release promotion or universal correctness claim. Token parity is the retained final transcript, not every task or sampling policy.','Cold section timing summarizes three resident requests after a cold model opening; coldOpening uses the actual first request separately.','Doppler full browser/process opening is unavailable; reported opening starts at model initialization.','Two independent cold openings per model and GPU, one per engine order; OS/driver caches not purged.','Different optimized formats: Doppler Q4_K/F32 compute/F16 KV versus ONNX q4f16.','No cross-engine dedicated GPU residency claim: reported memory surfaces differ.','Hosts were not exclusive. Browser versions differ between GPUs; comparisons are within each hardware lane.'])
(root/'comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['Cold rows below summarize resident requests after initialization; actual cold first requests are in the opening table. D = Doppler, T = Transformers.js.', '', '| GPU | Model | Mode / input | Order | Decode tok/s D / T | TTFT ms D / T | Complete ms D / T | Delivery p95 ms D / T | Tokens |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
for r in rows:
 a,b=r['doppler'],r['transformersjs'];pair=lambda k:f"{a[k]:.1f} / {b[k]:.1f}"
 lines.append(f"| {r['lane']} | {r['model'].removeprefix('qwen-3-5-').removesuffix('-q4k-ehaf16')} | {r['mode']} / {r['promptTokens']} | {r['order']} | {pair('decodeTokensPerSec')} | {pair('firstTokenMs')} | {pair('totalRunMs')} | {pair('decodeMsPerTokenP95')} | {r['outputParity']['status']} |")
lines.extend(['', '| GPU | Model | Order | Load ms D / T | Cold first request TTFT ms D / T | Opening to first output ms D / T | Opening to complete ms D / T |', '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
for r in rows:
 if not r['coldOpening']:continue
 a,b=r['coldOpening']['doppler'],r['coldOpening']['transformersjs'];pair=lambda k:f"{a[k]:.1f} / {b[k]:.1f}"
 lines.append(f"| {r['lane']} | {r['model'].removeprefix('qwen-3-5-').removesuffix('-q4k-ehaf16')} | {r['order']} | {pair('loadMs')} | {pair('firstTokenMs')} | {pair('firstResponseMs')} | {pair('completeMs')} |")
(root/'comparison-table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(receipts=len(rows),attempts=len(attempts),complete=summary['complete'],claimGrade=sum(r['fairness']['claimGrade'] for r in rows))))
if '--require-complete' in sys.argv and not summary['complete']:raise SystemExit(1)
