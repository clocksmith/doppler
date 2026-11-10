"""Check short-input prefill under the frozen policy; decode finiteness is separate.

Usage: compare-short.py PROTOCOL PREPARATION SHORT_RUN OBSERVATION REFERENCE POLICY NEW_OUTPUT
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

protocol_path, prep_path, run_path, observation_path, reference_path, policy_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
p, prep, run, obs, ref, policies = [json.loads(path.read_bytes()) for path in [
    protocol_path, prep_path, run_path, observation_path, reference_path, policy_path]]
assert obs['completed'] and not obs['gpuErrors'] and not obs['cleanupFailures']
assert not obs.get('routingControl') and not obs.get('counterfactual')
assert obs['runSha256'] == sha(run_path.read_bytes())
assert run['parentRunSha256'] == prep['runSha256']
assert obs['referenceSha256'] == sha(reference_path.read_bytes()) == p['bindings']['referenceSha256']
assert sha(policy_path.read_bytes()) == p['bindings']['policySha256']
assert obs['manifestSha256'] == p['bindings']['manifestSha256']
assert obs['inputsSha256'] == p['bindings']['inputsSha256']
assert len(obs['rows']) == 1
row = obs['rows'][0]
assert row['id'] == 'short' and row['logitRows'] == 3
expected = ref['sequences']['short']
file = reference_path.parent / expected['logitsFile']
assert sha(file.read_bytes()) == expected['logitsSha256']
source = np.load(file, allow_pickle=False)[expected['samplePositions'].index(row['inputTokenCount'] - 1)].astype('f8')
raw = Path(row['prefillLogitsFile']).read_bytes()
assert sha(raw) == row['prefillLogitsSha256']
actual = np.frombuffer(raw, '<f4').astype('f8')
assert source.shape == actual.shape == (248320,)
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = policies['tolerancePolicies'][policy_id]
delta = np.abs(source - actual)
violations = int(np.count_nonzero(delta > policy['absolute'] + policy['relative'] * np.abs(source)))
frames = []
for frame in row['logitFrames']:
    raw = Path(frame['file']).read_bytes()
    assert sha(raw) == frame['sha256']
    values = np.frombuffer(raw, '<f4')
    assert values.shape == (248320,) and np.isfinite(values).all()
    frames.append(dict(step=frame['step'], sha256=frame['sha256'], finiteValues=values.size))
result = dict(scope='Short-input independent prefill comparison plus three finite generation rows; no independent decode-logit parity or answer-quality claim',
              normalModelQualified=False, acceptanceChanged=False, observationSha256=sha(observation_path.read_bytes()),
              preparationSha256=sha(prep_path.read_bytes()), runSha256=sha(run_path.read_bytes()),
              referenceSha256=p['bindings']['referenceSha256'], policySha256=p['bindings']['policySha256'],
              producerSha256=sha(Path(__file__).read_bytes()), inputTokens=row['inputTokenCount'],
              prefillLogitsSha256=row['prefillLogitsSha256'], maxAbsoluteError=float(delta.max()),
              outsideUnchangedPolicy=violations, thisPrefillCheckPassed=violations == 0,
              firstTokenAgrees=bool(actual.argmax() == source.argmax()), frames=frames,
              tokenIds=row['tokenIds'], gpuErrors=obs['gpuErrors'], cleanupFailures=obs['cleanupFailures'])
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['maxAbsoluteError', 'outsideUnchangedPolicy', 'firstTokenAgrees']}))
