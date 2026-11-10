"""Bind committed decode evidence to its retained observation and source reference.

Usage: seal-receipt.py REPOSITORY CODE_REVISION GPU_OBSERVATION DECODE_REFERENCE NEW_RECEIPT
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import sys

repo, revision, observation_path, reference_path, output = sys.argv[1:]
repo, observation_path, reference_path, output = map(Path, [repo, observation_path, reference_path, output])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo)
commit = git('rev-parse', revision + '^{commit}').decode().strip()
directory = 'reports/flash-next-decode-20261110'
owner = 'artifacts/qwen38-flash-next-20261013'
def committed(file):
    return git('show', commit + ':' + file)
def report(name):
    return json.loads(committed(directory + '/' + name))
require(sha(committed(directory + '/seal-receipt.py')) == sha(Path(__file__).read_bytes()), 'Uncommitted receipt producer')
comparison, protocol, inputs = [report(name) for name in ['comparison.json', 'protocol.json', 'inputs-receipt.json']]
observation_raw, reference_raw = observation_path.read_bytes(), reference_path.read_bytes()
observation, reference = json.loads(observation_raw), json.loads(reference_raw)
require(sha(observation_raw) == comparison['gpuSha256'], 'GPU observation hash mismatch')
require(sha(reference_raw) == comparison['decodeReferenceSha256'] == protocol['decodeReferenceSha256'], 'Decode reference hash mismatch')
require(sha(committed(directory + '/protocol.json')) == comparison['protocolSha256'], 'Protocol hash mismatch')
require(sha(committed(directory + '/inputs-receipt.json')) == protocol['inputReceiptSha256'], 'Input preparation hash mismatch')
require(sha(committed(directory + '/reference-inputs.json')) == inputs['inputsSha256'] == protocol['decodeInputsSha256'], 'Decode input hash mismatch')
for name, digest in [('compare-learned-reference.py', comparison['comparatorSha256']),
                     ('prepare-learned-decode-inputs.py', inputs['producerSha256']),
                     ('prepare-learned-decode-run.py', protocol['producerSha256']),
                     ('prepare-learned-forward-reference.py', protocol['sourceProducerSha256'])]:
    require(sha(committed(owner + '/' + name)) == digest, 'Committed producer mismatch: ' + name)
require(observation['completed'] and not observation.get('failure') and not observation['gpuErrors']
        and not observation['cleanupFailures'], 'Execution did not complete cleanly')
require(not observation.get('routingControl') and not observation.get('counterfactual'), 'Substituted execution cannot qualify')
require(comparison['passed'] and comparison['kvStorageDtype'] == 'f32'
        and comparison['independentDecodeRowsChecked'] == comparison['decodeRowsRequested'] == 4, 'Decode gate did not pass completely')
for key in ['manifestSha256', 'inputsSha256', 'runSha256']:
    require(observation[key] == protocol[key], 'Observation identity mismatch: ' + key)
require(observation['referenceSha256'] == comparison['referenceSha256'] == protocol['originalReferenceSha256'], 'Original reference mismatch')
require(reference['producerSha256'] == protocol['sourceProducerSha256'] and reference['dtype'] == 'float32'
        and reference['surface'] == 'CPU' and reference.get('kvStorageDtype', 'f32') == 'f32', 'Wrong source lane')
require(reference['inputsSha256'] == protocol['decodeInputsSha256']
        and reference['artifactManifestSha256'] == protocol['manifestSha256'], 'Source identity mismatch')
policy_file = 'src/config/evidence/boundary-evidence-policy.json'
require(sha(committed(policy_file)) == comparison['policySha256'] == protocol['policySha256'], 'Committed policy mismatch')
require(not comparison['acceptanceChanged'] and not protocol['acceptanceChanged']
        and not inputs['acceptanceChanged'] and protocol['sourceGreedyChainVerifiedBeforeGpu'], 'Acceptance or input preparation changed')
require(comparison['policy'] == protocol['tolerancePolicy'], 'Policy values mismatch')
digests = committed('src/config/kernels/kernel-ref-digests.js').decode()
shaders = []
for name, entry in [('gated_delta_recurrent_sigmoid_gate.wgsl', 'main'),
                    ('gated_delta_fused_decode_sigmoid_gate.wgsl', 'main'),
                    ('scatter_add_routes_f32.wgsl', 'scatter_add_routes')]:
    file = 'src/gpu/kernels/' + name
    source = committed(file)
    require(sha(source) == observation['runtimeSourceSha256'][file], 'Committed targeted shader mismatch: ' + file)
    digest = sha((source.decode().replace('\r\n', '\n') + '\n@@entry:' + entry).encode())
    require(re.findall('"' + re.escape(name + '#' + entry) + r'":\s*"([a-f0-9]{64})"', digests) == [digest], 'Shader entry digest mismatch')
    shaders.append(dict(file=file, entryPoint=entry, sourceSha256=sha(source), contentDigest='sha256:' + digest,
                        gitBlob=git('rev-parse', commit + ':' + file).decode().strip()))
rows = []
require([r['id'] for r in observation['rows']] == [c['id'] for c in protocol['chains']] == ['short', 'long'], 'Prompt coverage mismatch')
for row, expected, checked in zip(observation['rows'], protocol['chains'], comparison['rows']):
    require(row['id'] == checked['id'] and row['inputTokenCount'] == expected['promptTokens']
            and row['kvStorageDtype'] == 'f32' and row['tokenIds'] == expected['expectedTokenIds'] == checked['tokenIds'], 'Prompt geometry or trajectory mismatch')
    require(checked['passed'] and checked['trajectoryAgrees'] and checked['fullLogitsChecked'], 'Incomplete prompt comparison')
    frames = row['logitFrames']
    metrics = [checked['logits']] + checked['decodeRows']
    require([f['step'] for f in frames] == [0, 1, 2] and len(metrics) == 3, 'Frame coverage mismatch')
    require(frames[0]['sha256'] == row['prefillLogitsSha256'], 'Prefill/frame identity mismatch')
    for frame, metric in zip(frames, metrics):
        require(frame['shape'] == [248320] and metric['count'] == 248320 and metric['passed']
                and metric['outsidePolicy'] == 0 and metric['actualArgmax'] == metric['referenceArgmax']
                == row['tokenIds'][frame['step']], 'Incomplete score or greedy comparison')
    rows.append(dict(id=row['id'], promptTokens=row['inputTokenCount'], tokenIds=row['tokenIds'],
                     frames=[dict(step=f['step'], shape=f['shape'], sha256=f['sha256'], comparison=m)
                             for f, m in zip(frames, metrics)]))
parent = json.loads(committed('reports/flash-next-route-products-20261110/receipt.json'))
long_identical = observation['rows'][1]['prefillLogitsSha256'] == parent['prefillLogitsSha256']
require(long_identical, 'Long prefill changed after short/reset')
observed_sources = [dict(file=file, observedSha256=digest, committedSha256=sha(committed(file)))
                    for file, digest in sorted(observation['runtimeSourceSha256'].items())]
result = dict(schema='doppler.flash-next-bounded-decode-receipt/v1',
              createdAtUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(), codeRevision=commit,
              component='doppler.repository-tooling', intent='preserved',
              scope='Repaired frozen AMD Node F32-KV snapshot: original short/long prefill and two cached decode steps each',
              boundedDecodeCheckPassed=True, normalModelQualified=False, currentMainRuntimeQualified=False,
              acceptanceChanged=False, producerSha256=sha(Path(__file__).read_bytes()),
              observationSha256=sha(observation_raw), decodeReferenceSha256=sha(reference_raw),
              sourceMethod=comparison['decodeReferenceMethod'], sourceCheckpoint=reference['checkpoint'],
              sourceUpstream=reference['upstream'], sourceTorch=reference['torch'], sourceThreads=reference['threads'],
              runtimeRevision=observation['runtimeRevision'], node=observation['process']['node'],
              manifestSha256=protocol['manifestSha256'], originalReferenceSha256=protocol['originalReferenceSha256'],
              originalInputsSha256=protocol['inputsSha256'], runSha256=protocol['runSha256'],
              policyId=protocol['policyId'], policySha256=protocol['policySha256'], tolerancePolicy=protocol['tolerancePolicy'],
              committedTargetedShaders=shaders, observedRuntimeSources=observed_sources,
              sourceLogitPayloads=protocol['sourceLogitPayloads'], rows=rows, checkedValues=6 * 248320, violations=0,
              longPrefillByteIdenticalAfterShortAndReset=long_identical, gpuErrors=[], cleanupFailures=[],
              serviceActions=False, gpuMemorySettingChanges=False, runtimeChanges=False, deployment=False,
              remaining=['Different prompts and longer generation including QSA block completion',
                         'Installed package and intended release configuration', 'Original F16-KV lane', 'Other platforms'],
              evidence=[dict(file=directory + '/' + name, sha256=sha(committed(directory + '/' + name)))
                        for name in ['README.md', 'comparison.json', 'inputs-receipt.json', 'protocol.json',
                                     'reference-inputs.json', 'retained-f16-failure.json', 'seal-receipt.py']])
with output.open('x') as handle:
    handle.write(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(codeRevision=commit, checkedValues=result['checkedValues'], violations=0,
                      snapshotSourcesDifferingFromCommit=sum(s['observedSha256'] != s['committedSha256'] for s in observed_sources))))
