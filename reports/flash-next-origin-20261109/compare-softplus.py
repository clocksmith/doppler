"""Compare an explicitly identified candidate with the unchanged logit policy.

Usage: compare-softplus.py PROTOCOL REFERENCE CPU_ROUTING_TRACE POLICY
       CANDIDATE_PREPARATION CANDIDATE_OBSERVATION NEW_OUTPUT
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

protocol_path, reference_path, cpu_root, policy_path, preparation_path, observation_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
protocol, reference, cpu, policies, preparation, observation = [json.loads(path.read_bytes()) for path in [
    protocol_path, reference_path, cpu_root / 'receipt.json', policy_path, preparation_path, observation_path]]
bindings = protocol['bindings']
for path, key in [(reference_path, 'referenceSha256'), (cpu_root / 'receipt.json', 'cpuRoutingTraceReceiptSha256'),
                  (policy_path, 'policySha256')]:
    assert sha(path.read_bytes()) == bindings[key]
assert observation['completed'] and not observation['gpuErrors'] and not observation['cleanupFailures']
assert not observation.get('routingControl') and not observation.get('counterfactual')
assert observation['runSha256'] == preparation['runSha256']
assert observation['manifestSha256'] == reference['artifactManifestSha256'] == bindings['manifestSha256']
assert observation['inputsSha256'] == reference['inputsSha256'] == bindings['inputsSha256']
assert observation['referenceSha256'] == bindings['referenceSha256']
assert preparation['artifactAndReferenceUnchanged'] and preparation['candidateStatePrecision'] == 'f32'
for change in preparation['changes']:
    assert observation['runtimeSourceSha256'][change['file']] == change['afterSha256']
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = policies['tolerancePolicies'][policy_id]
assert len(observation['rows']) == 1
row = observation['rows'][0]
assert row['id'] == 'long' and row['inputTokenCount'] == 2304
assert row['logitRows'] == 1 and row['kvStorageDtype'] == 'f32'
expected = reference['sequences']['long']
file = Path(expected['logitsFile'])
if not file.is_absolute():
    file = reference_path.parent / file
assert sha(file.read_bytes()) == expected['logitsSha256']
source = np.load(file, allow_pickle=False)[expected['samplePositions'].index(2303)].astype('f8')
raw = Path(row['prefillLogitsFile']).read_bytes()
assert sha(raw) == row['prefillLogitsSha256']
actual = np.frombuffer(raw, '<f4').astype('f8')
assert actual.shape == source.shape == (248320,) and np.isfinite(actual).all() and np.isfinite(source).all()
delta = np.abs(actual - source)
limit = policy['absolute'] + policy['relative'] * np.abs(source)
bad = np.flatnonzero(delta > limit)
routes = []
for record in row['routerObservations']:
    layer = record['layer']
    raw = Path(record['logits']['file']).read_bytes()
    assert sha(raw) == record['logits']['sha256']
    logits = np.frombuffer(raw, '<f4').reshape(2304, 512)
    assert np.isfinite(logits).all()
    name = f'router-{layer}.npz'
    assert sha((cpu_root / name).read_bytes()) == cpu['files'][name]
    with np.load(cpu_root / name, allow_pickle=False) as data:
        source_experts = data['experts'].copy()
    # This is a ranked-logit diagnostic, not an observation of GPU top-k output.
    ranked = np.argsort(-logits, axis=1, kind='stable')[:, :11]
    selected = ranked[:, :10]
    positions = np.flatnonzero(np.any(np.sort(selected, axis=1) != np.sort(source_experts, axis=1), axis=1))
    ties = np.flatnonzero(logits[np.arange(2304), ranked[:, 9]] == logits[np.arange(2304), ranked[:, 10]])
    routes.append(dict(layer=layer, differingSets=positions.tolist(), cutoffTies=ties.tolist()))
assert [r['layer'] for r in routes] == list(range(48))
result = dict(schema='doppler.flash-next-softplus-comparison/v1',
    scope='Frozen 2304-token last-position F32-KV candidate comparison; not complete model qualification',
    normalModelQualified=False, acceptanceChanged=False, runtimeReferenceInjection=False,
    policyId=policy_id, policySha256=bindings['policySha256'],
    manifestSha256=bindings['manifestSha256'], inputsSha256=bindings['inputsSha256'],
    referenceSha256=bindings['referenceSha256'],
    observationSha256=sha(observation_path.read_bytes()), preparationSha256=sha(preparation_path.read_bytes()),
    comparatorSha256=sha(Path(__file__).read_bytes()),
    observedLogits=actual.size, maxAbsoluteError=float(delta.max()),
    differenceRms=float(np.sqrt(np.mean(delta * delta))), outsideUnchangedPolicy=int(bad.size),
    thisLogitCheckPassed=not bool(bad.size), firstTokenAgrees=bool(actual.argmax() == source.argmax()),
    violations=[dict(token=int(i), actual=float(actual[i]), reference=float(source[i]),
                     absoluteError=float(delta[i]), limit=float(limit[i])) for i in bad],
    routing=dict(scope='CPU stable top-k inferred from captured GPU logits; GPU expert-index buffers were not captured',
                 tokenLayerDecisions=2304 * 48,
                 differences=[r for r in routes if r['differingSets'] or r['cutoffTies']]))
assert not output.exists()
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: result[key] for key in ['maxAbsoluteError', 'differenceRms', 'outsideUnchangedPolicy', 'thisLogitCheckPassed']}))
