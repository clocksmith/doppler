import pathlib,json,subprocess,os,sys
snapshot=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2]);models=pathlib.Path(sys.argv[3])
for size in ['0-8b','2b']:
 d=json.loads((root/f'{size}-doppler-first/compare_latest.json').read_text());dest=root/f'{size}-lifecycle';dest.mkdir(exist_ok=True)
 cfg={'request':{'runtimeConfig':d['sections']['warm']['doppler']['request']['runtimeConfig']}};(dest/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 env={**os.environ,'DOPPLER_PREPARED_MODEL_DIR':str(models/d['dopplerModelId']),'DOPPLER_PREPARED_CONFIG':str(dest/'config.json')};argv=['node','tests/integration/linear-prepared-sequence-physical.test.js','--require-gpu']
 (dest/'command.json').write_text(json.dumps({'cwd':str(snapshot),'argv':argv,'environment':{k:v for k,v in env.items() if k.startswith('DOPPLER_PREPARED_')}},indent=2)+'\n')
 with (dest/'run.log').open('w') as log:
  try:r=subprocess.run(argv,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240);status={'exitCode':r.returncode}
  except subprocess.TimeoutExpired:status={'timeout':True}
 (dest/'status.json').write_text(json.dumps(status)+'\n');print(size,status,flush=True)
