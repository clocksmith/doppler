from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
repo=root.parents[1]
d=json.loads((root/'main-smoke.json').read_text());assert d['ok']
a=d['result']['metrics'];b=json.loads((root/'runs/code-0-candidate.json').read_text())['result']['metrics']
assert a['referenceTranscript']['tokens']['ids']==b['referenceTranscript']['tokens']['ids']
assert a['decodeBottleneck']['schemaVersion']==2 and a['decodeBottleneck']['bottleneckClass'] is None
files=['src/cli/cli-output.js','src/inference/browser-harness/decode-diagnostics.js','src/config/runtime/profiles/qwen-3-5-0-8b-vulkan-throughput.json']
v={'ok':True,'same64Tokens':True,'diagnosticSchemaVersion':2,'bottleneckClass':None,'scope':'Physical smoke of patched main; excluded from paired speed estimator','sourceSha256':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in files}}
(root/'main-smoke-verification.json').write_text(json.dumps(v,indent=2)+'\n');print('Patched-main smoke verified: same 64 tokens; diagnostic v2; no causal bottleneck assertion.')
