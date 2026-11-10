"""Prepare an isolated route-product rounding candidate without changing frozen evidence.

Usage: prepare-route-product-candidate.py PARENT_RUNTIME PARENT_RUN NEW_SHADER
       PHYSICAL_TEST NEW_RUNTIME NEW_DATA_DIRECTORY
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

if len(sys.argv) != 7:
    raise SystemExit(__doc__)
parent, parent_run, shader, test, runtime, data = map(lambda p: Path(p).resolve(), sys.argv[1:])
assert parent.is_dir() and (parent / 'node_modules').is_dir()
assert all(p.is_file() for p in [parent_run, shader, test])
runtime.mkdir()
data.mkdir()
for name in ['src', 'tests']:
    shutil.copytree(parent / name, runtime / name, symlinks=True)
for name in ['package.json', 'package-lock.json', 'AGENTS.md', 'CATSCAN.md', 'GOALS.md']:
    if (parent / name).is_file():
        shutil.copy2(parent / name, runtime / name)
(runtime / 'node_modules').symlink_to(parent / 'node_modules', target_is_directory=True)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
file = 'src/gpu/kernels/scatter_add_routes_f32.wgsl'
target = runtime / file
before, after = target.read_bytes(), shader.read_bytes()
assert before != after and b'rounded_products' not in before and b'var<workgroup> rounded_products' in after
target.write_bytes(after)
entry = 'scatter_add_routes'
old_digest = sha(before.decode().replace('\r\n', '\n').encode() + ('\n@@entry:' + entry).encode())
new_digest = sha(after.decode().replace('\r\n', '\n').encode() + ('\n@@entry:' + entry).encode())
digests = runtime / 'src/config/kernels/kernel-ref-digests.js'
original_digests = digests.read_bytes()
key = 'scatter_add_routes_f32.wgsl#' + entry
old_text = '"' + key + '": "' + old_digest + '"'
new_text = '"' + key + '": "' + new_digest + '"'
assert original_digests.decode().count(old_text) == 1
digests.write_text(original_digests.decode().replace(old_text, new_text))
shutil.copy2(test, runtime / 'tests/kernels/moe-route-product-rounding-physical.test.js')
run = json.loads(parent_run.read_bytes())
graph = run['candidate']['sourceExecution']
updated = []
def refresh(value):
    if isinstance(value, list):
        for item in value:
            refresh(item)
    elif isinstance(value, dict):
        if value.get('kernel') == target.name and value.get('entry', 'main') == entry and 'digest' in value:
            assert value['digest'] == 'sha256:' + old_digest
            value['digest'] = 'sha256:' + new_digest
            updated.append(key)
        for item in value.values():
            refresh(item)
refresh(graph)
identity = sha(json.dumps(dict(graph=graph, shaderSha256=sha(after)), sort_keys=True, separators=(',', ':')).encode())
run['runtimeConfig']['inference']['kernelPath']['id'] += '-route-products-' + identity[:12]
run['runtimeConfig']['inference']['kernelPath']['description'] = 'Explicit ordered route-product candidate; unchanged weights, KV and acceptance.'
run['runtimeRevision'] += '+route-products-' + identity[:12]
run['candidate'] = dict(kind='softplus-and-ordered-route-products', parentRunSha256=sha(parent_run.read_bytes()),
                        sourceExecution=graph, sourceAndGraphSha256=identity,
                        shaderChanges=[dict(file=file, beforeSha256=sha(before), afterSha256=sha(after))])
run['sourceFiles'] = sorted(set(run['sourceFiles'] + [file]))
run['outputPath'] = str(data / 'observation.json')
(data / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
receipt = dict(scope='Isolated ordinary-F32 route-product rounding candidate; no reference tensor injection',
               normalModelQualified=False, producerSha256=sha(Path(__file__).read_bytes()),
               parentRunSha256=sha(parent_run.read_bytes()), runSha256=sha((data / 'run.json').read_bytes()),
               sourceAndGraphSha256=identity, changes=run['candidate']['shaderChanges'],
               kernelRefDigest=dict(key=key, before=old_digest, after=new_digest),
               digestTable=dict(beforeSha256=sha(original_digests), afterSha256=sha(digests.read_bytes())),
               graphReferencesUpdated=len(updated), artifactAndReferenceUnchanged=True,
               candidateStatePrecision='f32', serviceChanges=False)
(data / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(prepared=True, sourceAndGraphSha256=identity, graphReferencesUpdated=len(updated))))
