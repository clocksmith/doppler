"""Compare retained layer-16 output, layer-17 input, and normalization."""
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
raw = (root / 'input17-captures.json.gz').read_bytes()
receipt = json.loads((root / 'input17-result.json').read_text())
assert hashlib.sha256(raw).hexdigest() == receipt['diagnostic']['export']['sha256']
for name, binding in [('independent-layers.npz', 'independent-layers.json'), ('reference-layer17.npz', 'reference-layer17.json')]:
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == json.loads((root / binding).read_text())['sha256']
layer16 = np.load(root / 'independent-layers.npz')['layer_16']
norm17 = np.load(root / 'reference-layer17.npz')['attn.post_input_norm']
rows = []
for row in json.loads(gzip.decompress(raw))['records']:
    assert (row['layer'], row['stage']) in [(16,'layer.out'),(17,'attn.input'),(17,'attn.post_input_norm')]
    assert row['shape'] == [184,1024]
    data = base64.b64decode(row['data'])
    assert len(data) == row['byteLength'] == 184 * 1024 * 8
    assert row['storageDigest'] == 'sha256:' + hashlib.sha256(data).hexdigest()
    actual = np.frombuffer(data,dtype='<f4').astype(np.float64).reshape(184,1024,2).sum(axis=-1)
    assert np.isfinite(actual).all()
    expected = norm17 if row['stage']=='attn.post_input_norm' else layer16
    error = np.abs(actual-expected)
    rows.append({'layer':row['layer'],'stage':row['stage'],'ordinal':row['ordinal'],
      'maxAbsoluteError':float(error.max()),'maxIndex':list(map(int,np.unravel_index(error.argmax(),error.shape))),
      'matchingRows':np.flatnonzero(error.max(axis=1)<2e-11).tolist(),
      'storageDigest':row['storageDigest']})
assert any(r['stage']=='attn.input' for r in rows), 'Input observation absent'
result = {'scope':'Diagnostic input-boundary attribution; unchanged full-model gate',
 'captureSha256':hashlib.sha256(raw).hexdigest(),'records':rows,'failure':receipt.get('failure'),
 'cleanupFailure':receipt.get('cleanupFailure'),'providerSettlement':receipt.get('providerSettlement')}
(root/'input17-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'records':[{**r,'matchingRows':len(r['matchingRows'])} for r in rows]},indent=2))
