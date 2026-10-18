import json,os,pathlib,signal,subprocess,sys,time
host,contract_path=sys.argv[1:]
contract=json.loads(pathlib.Path(contract_path).read_text());lane=contract['hosts'][host]
root=pathlib.Path(contract['reportRoot']);base=root/host/contract.get('outputDirectory','final');base.mkdir(parents=True,exist_ok=True)
wait=lane.get('waitForReceipt')
if wait:
 deadline=time.monotonic()+900
 while not pathlib.Path(wait).exists():
  if time.monotonic()>deadline:raise RuntimeError('Prior isolated process did not settle')
  time.sleep(1)
rows=[]
for case in lane['cases']:
 out=base/case['id'];out.mkdir(parents=True,exist_ok=False)
 command=['node',str(root/case.get('driver','browser-compare.mjs')),lane[case['engine']],
  case['modelDir'],str(root/case['config']),str(out/'receipt.json')]
 row={'case':case,'command':command,'started':time.time()};print('START',case['id'],flush=True)
 with (out/'run.log').open('w') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:row['returncode']=process.wait(timeout=contract['timeoutSeconds'])
  except subprocess.TimeoutExpired:
   row['timeoutSeconds']=contract['timeoutSeconds'];os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 row['elapsedSeconds']=time.time()-row['started'];rows.append(row)
 (base/'campaign-status.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE',case['id'],row['returncode'] if 'returncode' in row else 'timeout',flush=True)
