import pathlib,json,subprocess,sys,time
snapshot=pathlib.Path(sys.argv[1]); models=pathlib.Path(sys.argv[2]); relay=pathlib.Path(sys.argv[3]); browser=sys.argv[4]; out=pathlib.Path(sys.argv[5]); out.mkdir(parents=True,exist_ok=True)
for size in ['0-8b','2b']:
 for order in ['doppler-first','transformersjs-first']:
  name=f'{size}-{order}';dest=out/name;dest.mkdir(exist_ok=True)
  model=f'qwen-3-5-{size}-q4k-ehaf16'
  argv=['node','tools/compare-engines.js','--model-id',model,'--model-url',(models/model).as_uri(),'--tjs-version','4','--tjs-dtype','q4f16','--tjs-local-model-path',str(relay),'--prefill-tokens','64','--max-tokens','128','--temperature','0','--top-k','1','--top-p','1','--seed','0','--warmup','2','--runs','3','--mode','warm','--decode-profile','parity','--doppler-surface','browser','--load-mode','http','--browser-executable',browser,'--resource-telemetry','on','--resource-telemetry-samples','--save','--save-dir',str(dest),'--skip-matrix-update','--timeout-ms','240000','--json','--warm-engine-order',order]
  (dest/'command.json').write_text(json.dumps({'cwd':str(snapshot),'argv':argv},indent=2)+'\n')
  start=time.time();print(json.dumps({'run':name,'phase':'started'}),flush=True)
  with (dest/'run.log').open('w') as log:
   try:r=subprocess.run(argv,cwd=snapshot,stdout=log,stderr=subprocess.STDOUT,timeout=900);result={'exitCode':r.returncode}
   except subprocess.TimeoutExpired:result={'timeout':True}
  result.update(run=name,elapsedSeconds=time.time()-start,receiptExists=(dest/'compare_latest.json').exists());(dest/'status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
