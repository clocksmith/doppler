"""Bind committed shader identities to the retained unchanged-policy comparison.

Usage: seal-receipt.py REPOSITORY CODE_REVISION LONG_OBSERVATION SHORT_OBSERVATION NEW_RECEIPT
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import sys

repo, revision, long_path, short_path, output = sys.argv[1:]
repo, long_path, short_path, output = map(Path, [repo, long_path, short_path, output])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo)
commit = git('rev-parse', revision + '^{commit}').decode().strip()
directory = 'reports/flash-next-route-products-20261110'
def committed(file):
    return git('show', commit + ':' + file)
def report(name):
    return json.loads(committed(directory + '/' + name))
require(sha(committed(directory + '/seal-receipt.py')) == sha(Path(__file__).read_bytes()),
        'Receipt producer differs from its committed source')
long_raw, short_raw = long_path.read_bytes(), short_path.read_bytes()
long, short = json.loads(long_raw), json.loads(short_raw)
comparison, short_comparison = report('long-comparison.json'), report('short-comparison.json')
preparation, delta = report('preparation.json'), report('source-delta.json')
require(sha(long_raw) == comparison['observationSha256'], 'Long observation hash mismatch')
require(sha(short_raw) == short_comparison['observationSha256'], 'Short observation hash mismatch')
require(long['runSha256'] == preparation['runSha256'] == delta['runSha256'], 'Candidate run identity mismatch')
require(comparison['outsideUnchangedPolicy'] == 0 and comparison['thisLogitCheckPassed'], 'Long gate did not pass')
require(short_comparison['outsideUnchangedPolicy'] == 0 and short_comparison['thisPrefillCheckPassed'], 'Short prefill did not pass')
require(not comparison['acceptanceChanged'] and not short_comparison['acceptanceChanged'], 'Acceptance changed')
require(delta['runtimeConfigNumericsUnchanged'] and delta['artifactReferenceInputsAndSequencesUnchanged'], 'Candidate scope changed')
long_comparator = 'reports/flash-next-origin-20261109/compare-softplus.py'
short_comparator = 'reports/flash-next-origin-20261109/compare-short.py'
require(sha(committed(long_comparator)) == comparison['comparatorSha256'], 'Long comparator identity mismatch')
require(sha(committed(short_comparator)) == short_comparison['producerSha256'], 'Short comparator identity mismatch')
policy_file = 'src/config/evidence/boundary-evidence-policy.json'
require(sha(committed(policy_file)) == comparison['policySha256'], 'Frozen policy identity mismatch')
for observation in [long, short]:
    require(observation['completed'] and not observation['gpuErrors'] and not observation['cleanupFailures'], 'Execution did not complete cleanly')
    require(not observation.get('routingControl') and not observation.get('counterfactual'), 'Substituted execution cannot qualify')
    for key in ['manifestSha256', 'inputsSha256', 'referenceSha256']:
        require(observation[key] == comparison[key], 'Execution identity mismatch: ' + key)
digests_file = 'src/config/kernels/kernel-ref-digests.js'
digest_table = committed(digests_file).decode()
shaders = []
for name, entry in [('gated_delta_recurrent_sigmoid_gate.wgsl', 'main'),
                    ('gated_delta_fused_decode_sigmoid_gate.wgsl', 'main'),
                    ('scatter_add_routes_f32.wgsl', 'scatter_add_routes')]:
    file = 'src/gpu/kernels/' + name
    source = committed(file)
    require(sha(source) == long['runtimeSourceSha256'][file] == short['runtimeSourceSha256'][file], 'Committed shader mismatch: ' + file)
    digest = sha((source.decode().replace('\r\n', '\n') + '\n@@entry:' + entry).encode())
    matches = re.findall('"' + re.escape(name + '#' + entry) + r'":\s*"([a-f0-9]{64})"', digest_table)
    require(matches == [digest], 'Committed entry-point digest mismatch: ' + file)
    shaders.append(dict(file=file, entryPoint=entry, sourceSha256=sha(source), contentDigest='sha256:' + digest,
                        gitBlob=git('rev-parse', commit + ':' + file).decode().strip()))
row = long['rows'][0]
require(row['id'] == 'long' and row['inputTokenCount'] == 2304 and row['logitRows'] == 1 and row['kvStorageDtype'] == 'f32', 'Unexpected long-run geometry or precision')
result = dict(schema='doppler.flash-next-route-products-receipt/v1',
              createdAtUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(), codeRevision=commit,
              component='doppler.runtime-source.gpu.kernels', intent='preserved',
              scope='Targeted committed shader sources bound to the frozen-snapshot AMD Node F32-KV long-prefill comparison',
              namedLongPrefillFailureClosed=True, normalModelQualified=False, acceptanceChanged=False,
              producerSha256=sha(Path(__file__).read_bytes()), committedShaders=shaders,
              candidateSourceTreeSha256=delta['sourceTreeAfterSha256'],
              qualifiedSnapshotSourceDeltaSha256=sha(committed(directory + '/source-delta.json')),
              longObservationSha256=sha(long_raw), shortObservationSha256=sha(short_raw),
              manifestSha256=comparison['manifestSha256'], inputsSha256=comparison['inputsSha256'],
              referenceSha256=comparison['referenceSha256'], policyId=comparison['policyId'],
              policySha256=comparison['policySha256'], prefillLogitsSha256=row['prefillLogitsSha256'],
              comparatorIdentities=[dict(file=file, sha256=sha(committed(file)))
                                    for file in [long_comparator, short_comparator, policy_file]],
              longPrefill=dict(tokens=2304, layers=48, lastPosition=2303, observedLogits=248320,
                               maxAbsoluteError=comparison['maxAbsoluteError'], differenceRms=comparison['differenceRms'],
                               violations=0, passed=True),
              shortPrefill=dict(tokens=short_comparison['inputTokens'], violations=0,
                                maxAbsoluteError=short_comparison['maxAbsoluteError'], passed=True),
              generationRows=dict(count=len(short_comparison['frames']), finite=True, independentParity=False),
              gpuErrors=[], cleanupFailures=[], referenceTensorInjection=False,
              serviceActions=False, gpuMemorySettingChanges=False, deployment=False,
              remaining=['Independent decode-logit parity', 'Unseen-input qualification',
                         'Original F16-KV lane qualification', 'Installed-package and release qualification'],
              evidence=[dict(file=directory + '/' + name, sha256=sha(committed(directory + '/' + name)))
                        for name in ['README.md', 'long-comparison.json', 'short-comparison.json', 'preparation.json',
                                     'source-delta.json', 'router-margin.json', 'operator-metal.json', 'operator-amd.json',
                                     'operator-before-metal.json', 'operator-before-amd.json',
                                     'rejected-local-helper-amd.json', 'rejected-local-helper.wgsl',
                                     'route-executor-metal.json', 'route-browser-metal.json']])
with Path(output).open('x') as handle:
    handle.write(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(codeRevision=commit, committedShaderCount=len(shaders), unchangedLongPrefillGate='passed', violations=0)))
