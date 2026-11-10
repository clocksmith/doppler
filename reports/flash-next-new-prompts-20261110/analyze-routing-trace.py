"""Compare retained new-prompt GPU observations with source routing rows.

Usage: analyze-routing-trace.py REFERENCE GPU_TRACE CPU_TRACE_ROOT PRIMARY_OBSERVATION NEW_OUTPUT
The GPU ranking is inferred from captured logits, not observed expert-index buffers.
"""
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np

reference_path, gpu_path, cpu_root, primary_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
def verified(path, digest):
    raw = path.read_bytes()
    require(sha(raw) == digest, 'Tensor hash mismatch: ' + path.name)
    return raw
ref_raw, gpu_raw, primary_raw, cpu_raw = [p.read_bytes() for p in [reference_path, gpu_path, primary_path, cpu_root / 'receipt.json']]
reference, gpu, primary, cpu = map(json.loads, [ref_raw, gpu_raw, primary_raw, cpu_raw])
require(cpu['completed'] and cpu['samplesAndRouterMetadataMatchReference'] and cpu['kvStorageDtype'] == 'f32', 'Incomplete independent source trace')
require(cpu['referenceSha256'] == gpu['referenceSha256'] == primary['referenceSha256'] == sha(ref_raw), 'Reference identity mismatch')
for key in ['manifestSha256', 'inputsSha256']:
    require(cpu[key] == gpu[key] == primary[key], 'Trace identity mismatch: ' + key)
for observation in [gpu, primary]:
    require(observation['completed'] and not observation['gpuErrors'] and not observation['cleanupFailures'], 'Incomplete GPU execution')
    require(not observation.get('counterfactual') and not observation.get('routingControl'), 'Substituted execution is not a normal trace')
require(len(gpu['rows']) == 1 and gpu['rows'][0]['id'] == cpu['sequenceId'], 'Unexpected trace cases')
row = gpu['rows'][0]
original = next(r for r in primary['rows'] if r['id'] == row['id'])
source = reference['sequences'][row['id']]
require(row['kvStorageDtype'] == original['kvStorageDtype'] == 'f32', 'GPU precision mismatch')
require(verified(Path(row['prefillLogitsFile']), row['prefillLogitsSha256'])
        == verified(Path(original['prefillLogitsFile']), original['prefillLogitsSha256']), 'Observation changed final scores')
def metrics(actual, expected):
    require(actual.shape == expected.shape and actual.size and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Invalid compared tensor')
    a, b = actual.astype('f8'), expected.astype('f8')
    delta = a - b
    denom = float(np.sum(b * b))
    return dict(maxAbsoluteError=float(np.max(np.abs(delta))), differenceRms=float(np.sqrt(np.mean(delta * delta))),
                referenceRms=float(np.sqrt(np.mean(b * b))), relativeL2=float(np.sqrt(np.sum(delta * delta) / denom)) if denom else None)
routes = []
require([r['layer'] for r in row['routerObservations']] == list(range(48)), 'Incomplete router coverage')
for record in row['routerObservations']:
    layer = record['layer']
    name = f'router-{layer}.npz'
    with np.load(io.BytesIO(verified(cpu_root / name, cpu['files'][name])), allow_pickle=False) as archive:
        src_input, src_logits, experts = archive['input'], archive['logits'], archive['experts']
    tensors = {}
    for role in ['input', 'logits']:
        desc = record[role]
        tensors[role] = np.frombuffer(verified(Path(desc['file']), desc['sha256']), dtype='<f4').reshape(desc['shape'])
    ranked = np.argsort(-tensors['logits'], axis=1, kind='stable')[:, :experts.shape[1]]
    different = np.flatnonzero(np.any(np.sort(ranked, axis=1) != np.sort(experts, axis=1), axis=1))
    source_ranked = np.argsort(-src_logits, axis=1, kind='stable')[:, :experts.shape[1]]
    source_rank_different = np.flatnonzero(np.any(np.sort(source_ranked, axis=1) != np.sort(experts, axis=1), axis=1))
    margins = source['router'][str(layer)]['marginKthToNext']
    routes.append(dict(layer=layer, input=metrics(tensors['input'], src_input), logits=metrics(tensors['logits'], src_logits),
                       inferredSetDifferences=different.tolist(), sourceRawLogitRankingDiffersFromActualExperts=source_rank_different.tolist(),
                       sourceMinimumRawLogitMargin=min(margins), sourceMinimumMarginPosition=int(np.argmin(margins)),
                       inputSha256=record['input']['sha256'], logitsSha256=record['logits']['sha256']))
layers = []
for capture in row['layers']:
    raw = verified(Path(capture['file']), capture['sha256'])
    actual = np.frombuffer(raw, dtype='<f4').reshape(capture['shape'])
    positions = source['samplePositions']
    sampled = actual[positions, ::reference['sampleStride']]
    result = dict(layer=capture['layer'], capturedSha256=capture['sha256'], sampled=metrics(sampled, np.array(source['layers'][str(capture['layer'])]['samples'], dtype='<f4')))
    name = f"layer-{capture['layer']}.npy"
    if name in cpu['files']:
        expected = np.load(io.BytesIO(verified(cpu_root / name, cpu['files'][name])), allow_pickle=False)
        result['fullTensor'] = metrics(actual, expected.reshape(actual.shape))
    layers.append(result)
qsa = []
for record in row['qsaObservations']:
    for query in record['rows']:
        expected = source['qsa'][str(record['layer'])][str(query['position'])]
        if query['addedBlocks'] or query['removedBlocks'] or query['count'] != expected['selectedTokens']:
            qsa.append(dict(layer=record['layer'], position=query['position'], actualCount=query['count'], sourceCount=expected['selectedTokens'],
                            addedBlocks=query['addedBlocks'], removedBlocks=query['removedBlocks'], sourceBlockMargin=query['sourceBlockMargin']))
result = dict(schema='doppler.flash-next-new-prompt-routing-trace/v1',
              scope='All-position source router rows vs captured GPU logits and inputs; inferred GPU rankings are not observed routing choices',
              producerSha256=sha(Path(__file__).read_bytes()), referenceSha256=sha(ref_raw), gpuTraceSha256=sha(gpu_raw),
              primaryObservationSha256=sha(primary_raw), cpuTraceReceiptSha256=sha(cpu_raw),
              manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
              finalScoresByteIdenticalWithAndWithoutObservation=True, layers=layers, routes=routes, sampledQsaDifferences=qsa,
              firstInferredRoutingDifference=next((dict(layer=r['layer'], positions=r['inferredSetDifferences']) for r in routes if r['inferredSetDifferences']), None),
              normalModelQualified=False, acceptanceChanged=False)
with output.open('x') as handle:
    handle.write(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(firstInferredRoutingDifference=result['firstInferredRoutingDifference'], layers=layers)))
