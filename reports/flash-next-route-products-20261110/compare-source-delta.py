"""Verify the isolated source delta and unchanged numeric runtime configuration.

Usage: compare-source-delta.py PARENT_RUNTIME CANDIDATE_RUNTIME PARENT_RUN RUN NEW_OUTPUT
"""
from pathlib import Path
import hashlib
import json
import sys

parent, candidate, parent_run, run, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def inventory(root):
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}
before, after = inventory(parent / 'src'), inventory(candidate / 'src')
assert before.keys() == after.keys()
changes = [dict(file='src/' + name, beforeSha256=before[name], afterSha256=after[name])
           for name in before if before[name] != after[name]]
assert {r['file'] for r in changes} == {'src/gpu/kernels/scatter_add_routes_f32.wgsl',
                                     'src/config/kernels/kernel-ref-digests.js'}
a, b = json.loads(parent_run.read_bytes()), json.loads(run.read_bytes())
assert all(a[key] == b[key] for key in ['expectedManifestSha256', 'referencePath', 'inputsPath', 'sequences'])
for config in [a['runtimeConfig'], b['runtimeConfig']]:
    for key in ['id', 'description']:
        config['inference']['kernelPath'].pop(key)
assert a['runtimeConfig'] == b['runtimeConfig']
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
result = dict(scope='Complete source-tree and numeric runtime-config comparison, excluding kernel-path identity labels',
              producerSha256=sha(Path(__file__).read_bytes()), parentRunSha256=sha(parent_run.read_bytes()),
              runSha256=sha(run.read_bytes()), sourceFileCount=len(before), sameSourceFileInventory=True,
              sourceTreeBeforeSha256=sha(canonical(before)), sourceTreeAfterSha256=sha(canonical(after)),
              runtimeConfigNumericsUnchanged=True, numericRuntimeConfigSha256=sha(canonical(a['runtimeConfig'])),
              artifactReferenceInputsAndSequencesUnchanged=True, sourceChanges=changes)
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
