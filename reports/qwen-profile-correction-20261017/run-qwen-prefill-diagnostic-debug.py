import pathlib,json,subprocess,sys,time
snapshot=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); models=pathlib.Path(sys.argv[3])
for size in ['0-8b','2b']:
 source=root/f'{size}-doppler-first/compare_latest.json'
 d=json.loads(source.read_text());s=d['sections']['warm'];runtime=s['doppler']['request']['runtimeConfig']
 runtime['shared']['benchmark']['run'].update(maxNewTokens=1,warmupRuns=2,timedRuns=1,profile=True)
 runtime['shared']['debug']['profiler'].update(enabled=True)
 dest=root/f'{size}-prefill-diagnostic-debug';dest.mkdir(exist_ok=True)
 config={'request':{'workload':'inference','modelId':d['dopplerModelId'],'modelUrl':(models/d['dopplerModelId']).as_uri(),'inferenceInput':{'prompt':d['prompt'],'maxTokens':1},'runtimeConfig':runtime}}
 (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 argv=['node','src/cli/doppler-cli.js','debug','--surface','browser','--config',str(dest/'config.json'),'--json']
 (dest/'command.json').write_text(json.dumps({'cwd':str(snapshot),'argv':argv,'scope':'diagnostic GPU timestamps; same 64 input tokens; 1 output token; 2 warmup,1 observed; not comparative performance'},indent=2)+'\n')
 with (dest/'run.log').open('w') as f:
  try:r=subprocess.run(argv,cwd=snapshot,stdout=f,stderr=subprocess.STDOUT,timeout=240);status={'exitCode':r.returncode}
  except subprocess.TimeoutExpired:status={'timeout':True}
 (dest/'status.json').write_text(json.dumps(status)+'\n');print(size,status,flush=True)
