"""Fixed eight-pair protocol; never reuse an existing result path."""
import hashlib,json,pathlib,subprocess,time
p=pathlib.Path(__file__).resolve().parent
contract=json.loads((p/'contract.json').read_text())
reference=json.loads((p/'baseline-reproduction.json').read_text())['result']['rows']
expected=next(r for r in reference if r['stage']=='first')
logits=next(r for r in reference if r['stage']=='numerical')['logitsSha256']
roots={'baseline':'.cache/qwen-bind-group-baseline-20261017','candidate':'.cache/qwen-scratch-candidate-20261018'}
for pair,order in enumerate(contract['primary']['order']):
 for arm in order.split(','):
  out=p/f'pair-{pair:02d}-{arm}.json'
  assert not out.exists(),out
  started=time.time()
  with out.with_suffix('.log').open('w') as log:
   subprocess.run(['node',str(p/'run.js'),roots[arm],'.cache/qwen-2b-standard-blake3-20261017','reports/qwen-bind-group-reuse-20261017/metal-config.json',str(out),'measure'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
  result=json.loads(out.read_text());assert result['passed']
  for row in result['result']['rows']:
   if 'tokens' in row: assert row['tokens']==expected['tokens'] and row['text']==expected['text'],(out,row['stage'])
   if row['stage']=='numerical': assert row['logitsSha256']==logits,out
  print(json.dumps({'pair':pair,'arm':arm,'passed':True,'elapsedSeconds':time.time()-started}),flush=True)
