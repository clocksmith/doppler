"""Compare source/GPU stage slices before the first routing difference.

Usage: compare-upstream-stages.py CPU_SUFFIX_TRACE CPU_TARGET_TRACE GPU_TRACE ROUTING_TRACE MANIFEST VERIFIED_READER NEW_OUTPUT
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

suffix_path, target_path, gpu_path, routing_path, manifest_path, reader_path, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('verified_tensor_io', reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
paths = [suffix_path, target_path, gpu_path, routing_path, manifest_path]
raws = [p.read_bytes() for p in paths]
suffix, target, gpu, routing, manifest = map(json.loads, raws)
require(gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures'], 'GPU diagnostic incomplete')
for source in [suffix, target]:
    require(source['sourceRouterInputReplayBitExact'] and source['diagnosticCompleted'], 'Source trace not coupled to independent router inputs')
    require(source['manifestSha256'] == gpu['manifestSha256'] == sha(raws[-1])
            and source['inputsSha256'] == gpu['inputsSha256'] == routing['inputsSha256'], 'Artifact/input identity mismatch')
require(suffix['sequenceId'] == target['sequenceId'] == gpu['sequenceId'], 'Different source/GPU cases')
tokens = target['inputTokens']
positions = next(r['positions'] for r in target['tensors'] if r['role'] == 'mlp_hc' and r['stage'] == 'branchInput')
layer = target['layer']
cpu_index = {}
for source, directory in [(suffix, suffix_path.parent), (target, target_path.parent)]:
    records = source.get('sourceTensorsAcrossResumedLayers', source['tensors'])
    for record in records:
        key = (record.get('layer', source['layer']), record['role'], record['stage'])
        require(key not in cpu_index, 'Duplicate source boundary')
        cpu_index[key] = (record, directory)
gpu_records = list(gpu['tensors'])
gpu_records.append(dict(gpu['attentionProjectionInput'], layer=layer, role='attention', stage='projectionInput', positions=None))
gpu_index = {(r['layer'], r['role'], r['stage']): r for r in gpu_records}
require(len(gpu_index) == len(gpu_records), 'Duplicate GPU boundary')
def selected(record, directory):
    values = reader.read_tensor(record, directory)
    if record.get('positions') is not None:
        require(record['positions'] == positions and values.shape[0] == len(positions), 'Different slice positions')
        return values.reshape(len(positions), -1)
    require(values.shape[0] == tokens, 'Different full tensor token axis')
    return values.reshape(tokens, -1)[positions]
key = (layer, 'mlp_hc', 'branchInput')
observed = selected(gpu_index[key], gpu_path.parent)
record = next(r for r in routing['rows'][0]['routerObservations'] if r['layer'] == layer)['input']
original = reader.read_tensor(record, routing_path.parent)[positions]
require(np.array_equal(observed.view('u4'), original.view('u4')), 'New stage observer changed the target router input')
rows = []
def metrics(actual, expected):
    require(actual.shape == expected.shape, 'Different stage shape')
    a, b = actual.astype('f8'), expected.astype('f8')
    delta = a - b
    denominator = float(np.sum(b * b))
    return dict(values=int(actual.size), referenceRms=float(np.sqrt(np.mean(b * b))),
                maxAbsoluteError=float(np.max(np.abs(delta))), differenceRms=float(np.sqrt(np.mean(delta * delta))),
                relativeL2=float(np.sqrt(np.sum(delta * delta) / denominator)) if denominator else None)
for key in sorted(set(cpu_index) & set(gpu_index)):
    source, directory = cpu_index[key]
    actual = selected(gpu_index[key], gpu_path.parent)
    expected = selected(source, directory)
    per_position = []
    for index, position in enumerate(positions):
        entry = dict(position=position, **metrics(actual[index], expected[index]))
        if key[1] in ['attn_hc', 'mlp_hc'] and key[2] in ['input', 'norm']:
            streams = manifest['inference']['hyperConnections']['streamCount']
            width = manifest['architecture']['hiddenSize']
            require(actual.shape[1] == streams * width, 'HC stream geometry mismatch')
            a, b = actual[index].reshape(streams, width), expected[index].reshape(streams, width)
            entry['streams'] = [dict(stream=s, **metrics(a[s], b[s])) for s in range(streams)]
        per_position.append(entry)
    rows.append(dict(layer=key[0], role=key[1], stage=key[2], positions=positions,
                     sourceSha256=source['sha256'], gpuSha256=gpu_index[key]['sha256'],
                     **metrics(actual, expected), perPosition=per_position))
result = dict(schema='doppler.flash-next-new-prompt-upstream-stages/v1',
              scope='Matched source/GPU stage slices; source and GPU router-input replay controls are bit-exact to their own retained traces',
              producerSha256=sha(Path(__file__).read_bytes()), readerSha256=sha(reader_path.read_bytes()),
              identities=[dict(file=p.name, sha256=sha(raw)) for p, raw in zip(paths, raws)],
              sourceRouterInputReplayBitExact=True, gpuRouterInputReplayBitExact=True, positions=positions, stages=rows,
              unmatchedSourceBoundaries=[list(k) for k in sorted(set(cpu_index) - set(gpu_index))],
              unmatchedGpuBoundaries=[list(k) for k in sorted(set(gpu_index) - set(cpu_index))],
              normalModelQualified=False, acceptanceChanged=False)
with output.open('x') as handle:
    handle.write(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(stages=len(rows), positions=positions, gpuRouterInputReplayBitExact=True)))
