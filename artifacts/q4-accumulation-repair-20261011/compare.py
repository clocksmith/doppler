import json,base64,numpy as np
from pathlib import Path
root=Path(__file__).parent
control=json.load(open('reports/local/whole-model-f64-20261011/comparison.json'))
candidate=json.load(open(root/'pinned-candidate-prefixes.json'))
previous=json.load(open('reports/local/accumulation-audit-20261011/tiled-candidate-prefixes.json'))
rows=[]
for expected,old,new in zip(control['results'],previous['results'],candidate['results'],strict=True):
 assert expected['step']==old['step']==new['step']
 ideal=np.frombuffer(base64.b64decode(expected['logitsF64']),dtype='<f8')
 row={'step':expected['step'],'expectedToken':expected['greedyToken']}
 for label,value in [('tiledOnly',old),('tiledPlusQ4',new)]:
  actual=np.frombuffer(base64.b64decode(value['logits']),dtype='<f4').astype(np.float64)
  error=np.abs(actual-ideal)
  row[label]={'maxAbsError':float(error.max()),'rmsError':float(np.sqrt(np.mean(error**2))),'valuesOverTolerance':int(np.sum(error>0.001)),'token':int(actual.argmax())}
 rows.append(row)
report={'tolerance':0.001,'qualified':False,'rows':rows,'quantizedDispatch':candidate['quantizedDispatch']}
(root/'model-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(rows,indent=2))
