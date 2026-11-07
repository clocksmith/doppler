"""Compare a separate F32-KV observation under the unchanged quantized-v1 policy.

Usage: python3 compare-f32kv-reference.py GPU REFERENCE REFERENCE_RUN OUTPUT
The original F16-KV reference and its failing verdict are never changed.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flash-next-qualification-20261106'))
from verified_tensor_io import read_tensor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(gpu_path, reference_path, binding_path, output):
    gpu_path, reference_path, binding_path, output = map(Path, [gpu_path, reference_path, binding_path, output])
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    gpu_bytes, ref_bytes, binding_bytes = [p.read_bytes() for p in [gpu_path, reference_path, binding_path]]
    gpu, ref, binding = map(json.loads, [gpu_bytes, ref_bytes, binding_bytes])
    require(ref['schema'] == 'doppler.qwen38-learned-forward-reference/v1', 'Unexpected reference schema')
    require(gpu.get('completed') and not gpu.get('failure'), 'Incomplete GPU observation')
    require(not gpu['gpuErrors'] and not gpu['cleanupFailures'] and not gpu.get('routingControl'), 'Failed or counterfactual GPU observation')
    require(gpu['runtimeConfig']['inference']['session']['kvcache']['kvDtype'] == 'f32', 'GPU run does not declare F32 KV')
    require(binding['kvStorageDtype'] == 'f32' and ref['dtype'] == 'float32', 'Reference must declare F32 execution')
    args = binding['argv']
    require('--kv-dtype' not in args, 'This unchanged producer uses native F32 storage only without the F16 hook')
    require(sha(Path(args[0]).read_bytes()) == binding['producerSha256'] == ref['producerSha256'], 'Reference producer drift')
    require(ref['artifactManifestSha256'] == binding['manifestSha256'] == gpu['manifestSha256'], 'Different artifact weights')
    require(ref['inputsSha256'] == binding['inputsSha256'] == gpu['inputsSha256'], 'Different input identities')
    inputs = json.loads(Path(args[args.index('--inputs')+1]).read_bytes())
    require(sha(Path(args[args.index('--inputs')+1]).read_bytes()) == ref['inputsSha256'], 'Input bytes drifted')
    manifest_path = Path(args[args.index('--artifact')+1]) / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    require(sha(manifest_bytes) == gpu['manifestSha256'], 'Artifact manifest drifted')
    vocab = json.loads(manifest_bytes)['architecture']['vocabSize']
    policy_path = Path(__file__).resolve().parents[2] / 'src/config/evidence/boundary-evidence-policy.json'
    policy_bytes = policy_path.read_bytes()
    policy_id = 'doppler.boundary-tolerance/quantized-v1'
    policy = json.loads(policy_bytes)['tolerancePolicies'][policy_id]
    rows = []
    for row in gpu['rows']:
        require(row['kvStorageDtype'] == 'f32', 'Observed cache does not use F32 storage')
        expected = ref['sequences'][row['id']]
        ids = next(s['tokens'] for s in inputs['sequences'] if s['id'] == row['id'])
        require(len(ids) == row['inputTokenCount'] == expected['tokens'], 'Sequence geometry differs')
        actual = read_tensor(dict(file=row['prefillLogitsFile'], sha256=row['prefillLogitsSha256'], shape=[vocab])).astype('f8')
        source = read_tensor(dict(file=expected['logitsFile'], sha256=expected['logitsSha256'],
                                 shape=[len(expected['samplePositions']), vocab]), reference_path.parent)
        source = source[expected['samplePositions'].index(len(ids)-1)].astype('f8')
        delta = np.abs(actual-source)
        outside = int(np.sum(delta > policy['absolute']+policy['relative']*np.abs(source)))
        argmax = int(np.argmax(actual))
        matches = argmax == expected['argmax'][len(ids)-1] == row['tokenId']
        log_p = source-(np.max(source)+np.log(np.sum(np.exp(source-np.max(source)))))
        log_q = actual-(np.max(actual)+np.log(np.sum(np.exp(actual-np.max(actual)))))
        top_p, top_q = np.argsort(source)[-20:], np.argsort(actual)[-20:]
        rows.append(dict(id=row['id'], count=vocab, maxAbsoluteError=float(delta.max()),
                         rmsError=float(np.sqrt(np.mean(delta*delta))), outsidePolicy=outside,
                         firstTokenAgrees=bool(matches), passed=outside == 0 and bool(matches),
                         diagnostics=dict(klNats=float(np.sum(np.exp(log_p)*(log_p-log_q))),
                                          top20Overlap=len(set(top_p)&set(top_q)))))
    require(rows, 'No observed rows')
    report = dict(scope='Separate F32-KV candidate against matching independent source; original F16 gate untouched',
                  policyId=policy_id, policySha256=sha(policy_bytes), gpuSha256=sha(gpu_bytes),
                  referenceSha256=sha(ref_bytes), referenceBindingSha256=sha(binding_bytes),
                  manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
                  rows=rows, passed=all(r['passed'] for r in rows),
                  scopeOfVerdict='Observed prefill rows only; decode and full-model qualification are not established',
                  fullModelQualified=False,
                  diagnosticsAreNotAcceptance=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
