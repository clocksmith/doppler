import hashlib,json,pathlib,subprocess,sys,time
root,models,relay=map(pathlib.Path,sys.argv[1:4]);browser=sys.argv[4];out=pathlib.Path(sys.argv[5]);inventory=pathlib.Path(sys.argv[6]);out.mkdir(parents=True,exist_ok=True)
archive_sha='75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2'
contract=dict(archiveSha256=archive_sha,models=['qwen-3-5-0-8b-q4k-ehaf16','qwen-3-5-2b-q4k-ehaf16'],formats={'doppler':'Q4_K weights, F32 compute, F16 KV','transformersjs':'ONNX q4f16; formats differ'},workloads=[dict(mode='warm',promptTokens=64,decodeTokens=128,warmup=2,runs=3),dict(mode='cold',promptTokens=64,decodeTokens=128,warmup=0,runs=3),dict(mode='warm',promptTokens=1024,decodeTokens=128,warmup=2,runs=3)],orders=['doppler-first','transformersjs-first'],sampling=dict(temperature=0,topK=1,topP=1,seed=0),coldScope='engine profile/cache cold; provisioned local artifacts; OS and driver caches not purged; not network-cold',acceptance='existing comparator prompt/token/sampling/fairness gates unchanged; keep failed and non-comparable runs',memoryScope='retain engine/process/device telemetry with its provenance; unequal memory surfaces cannot establish a cross-engine VRAM win')
(out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
records=json.loads(inventory.read_text());statuses=[]
for mode,prefill,warmup in [('warm',64,2),('cold',64,0),('warm',1024,2)]:
 for size in ['0-8b','2b']:
  for order in contract['orders']:
   name=f'{mode}-p{prefill}-{size}-{order}';dest=out/name;dest.mkdir(exist_ok=False)
   bad=[r['path'] for r in records if hashlib.sha256((root/r['path']).read_bytes()).hexdigest()!=r['sha256']]
   if bad:raise RuntimeError(f'Frozen runtime drift: {bad}')
   model=f'qwen-3-5-{size}-q4k-ehaf16'
   argv=['node','tools/compare-engines.js','--model-id',model,'--model-url',(models/model).as_uri(),'--tjs-version','4','--tjs-dtype','q4f16','--tjs-local-model-path',str(relay),'--prefill-tokens',str(prefill),'--max-tokens','128','--temperature','0','--top-k','1','--top-p','1','--seed','0','--warmup',str(warmup),'--runs','3','--mode',mode,'--decode-profile','parity','--doppler-surface','browser','--load-mode','http','--browser-executable',browser,'--resource-telemetry','on','--resource-telemetry-samples','--save','--save-dir',str(dest),'--skip-matrix-update','--timeout-ms','240000','--json','--warm-engine-order',order,'--cold-engine-order',order]
   command=dict(cwd=str(root),argv=argv,archiveSha256=archive_sha,verifiedArchiveFiles=len(records),comparatorDriverSha256=hashlib.sha256((root/'tools/compare-engines.js').read_bytes()).hexdigest(),modelManifestSha256=hashlib.sha256((models/model/'manifest.json').read_bytes()).hexdigest())
   (dest/'command.json').write_text(json.dumps(command,indent=2)+'\n');start=time.monotonic();print(json.dumps(dict(run=name,phase='started')),flush=True)
   with (dest/'run.log').open('w') as log:
    try:p=subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=900);status=dict(exitCode=p.returncode)
    except subprocess.TimeoutExpired:status=dict(timeout=True)
   status.update(run=name,elapsedSeconds=time.monotonic()-start,receiptExists=(dest/'compare_latest.json').exists());statuses.append(status)
   (dest/'status.json').write_text(json.dumps(status,indent=2)+'\n');(out/'status.json').write_text(json.dumps(statuses,indent=2)+'\n');print(json.dumps(status),flush=True)
   if status.get('exitCode',1)!=0 and not status['receiptExists']:raise RuntimeError('Comparator setup failed; repair before remaining configurations')

