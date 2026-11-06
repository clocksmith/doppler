"""Separate propagated input drift from local grouped-norm arithmetic error.

Usage: python3 -P check-hc-norm-same-input.py <gpu-trace.json> <cpu-trace.json>
       <artifact> <reference-data> <layer> <position> <out.json>
Uses the unchanged pinned CPU operator; no tensor feedback or acceptance change.
"""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

gpu_path, cpu_path, artifact, data = map(Path, sys.argv[1:5])
layer, position = map(int, sys.argv[5:7])
output = Path(sys.argv[7])
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes, cpu_bytes, manifest_bytes = gpu_path.read_bytes(), cpu_path.read_bytes(), (artifact / 'manifest.json').read_bytes()
gpu, cpu, manifest = json.loads(gpu_bytes), json.loads(cpu_bytes), json.loads(manifest_bytes)
assert gpu['diagnosticCompleted'] and cpu['diagnosticCompleted']
assert not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['manifestSha256'] == cpu['manifestSha256'] == sha(manifest_bytes)
assert gpu['inputsSha256'] == cpu['inputsSha256']
geometry = manifest['inference']['hyperConnections']
binding = json.loads((data / 'source-binding.json').read_bytes())
source = (data / 'source/modeling_qwen4_exp.py').read_bytes()
entry = next(r for r in binding['files'] + binding['referenceImplementation']['files'] if r['path'] == 'modeling_qwen4_exp.py')
assert sha(source) == entry['sha256']
node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextRMSNorm')
assert not node.decorator_list
namespace = dict(nn=torch.nn, torch=torch)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned-Qwen4ExpTextRMSNorm', 'exec'), namespace)
spec = importlib.util.spec_from_file_location('stage_metrics', Path(__file__).with_name('compare-upstream-stages.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
torch.set_num_threads(16)
report = dict(scope='Identical-input normalization versus propagated source-input drift; not model acceptance',
              normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
              manifestSha256=sha(manifest_bytes), sourceSha256=sha(source),
              scriptSha256=sha(Path(__file__).read_bytes()), torch=torch.__version__, threads=torch.get_num_threads(),
              layer=layer, position=position, rows=[])

def read_stage(trace, role, stage, device):
    record = next(t for t in trace['tensors'] if t['role'] == role and t['stage'] == stage
                  and (device == 'cpu' or t['layer'] == layer))
    file = Path(record['file']) if device == 'gpu' else cpu_path.parent / record['file']
    raw = file.read_bytes()
    assert sha(raw) == record['sha256'] and record['positions'] == [position]
    values = np.frombuffer(raw, dtype='<f4').reshape(record['shape']) if device == 'gpu' else np.load(file, allow_pickle=False)
    return values, record


for role, tensor_prefix in [('attn_hc', 'attn_hyper_connection'), ('mlp_hc', 'mlp_hyper_connection')]:
    gx, record = read_stage(gpu, role, 'input', 'gpu')
    gy, _ = read_stage(gpu, role, 'norm', 'gpu')
    cx, _ = read_stage(cpu, role, 'input', 'cpu')
    cy, _ = read_stage(cpu, role, 'norm', 'cpu')
    descriptor = manifest['tensors'][f'model.language_model.layers.{layer}.{tensor_prefix}.hc_norm.weight']
    assert descriptor['dtype'] == 'F16'
    raw = bytearray()
    for span in descriptor.get('spans') or [dict(shardIndex=descriptor['shard'], offset=descriptor['offset'], size=descriptor['size'])]:
        with (artifact / manifest['shards'][span['shardIndex']]['filename']).open('rb') as handle:
            handle.seek(span['offset'])
            part = handle.read(span['size'])
            assert len(part) == span['size']
            raw.extend(part)
    weights = np.frombuffer(raw, dtype='<f2').astype(np.float32)
    width = geometry['streamCount'] * geometry['streamWidth']
    assert weights.shape == (width,) and gx.shape == cx.shape == (1, width)
    norm = namespace['Qwen4ExpTextRMSNorm'](width, geometry['streamWidth'], geometry['normEps'])
    shape = record['originalShape']
    assert shape[1] == width and 0 <= position < shape[0]
    with torch.no_grad():
        norm.weight.copy_(torch.from_numpy(weights))
        batch = torch.zeros(shape, dtype=torch.float32)
        batch[position].copy_(torch.from_numpy(cx[0].copy()))
        replay = norm(batch)[position].numpy().copy()[None, :]
        assert np.array_equal(replay, cy), 'exact CPU input must reproduce the original source output bit-for-bit'
        batch[position].copy_(torch.from_numpy(gx[0].copy()))
        source_on_gpu_input = norm(batch)[position].numpy().copy()[None, :]
    total = gy.astype(np.float64) - cy
    local = gy.astype(np.float64) - source_on_gpu_input
    propagated = source_on_gpu_input.astype(np.float64) - cy
    assert np.array_equal(local + propagated, total)
    measure = lambda a, b: metrics.stage_metrics(a, b, [position], geometry['streamCount'], geometry['streamWidth'])
    report['rows'].append(dict(role=role, geometryReplayBitExact=True, weightSha256=sha(raw),
        input=measure(gx, cx), total=measure(gy, cy),
        localOnSameGpuInput=measure(gy, source_on_gpu_input), propagatedSourceInput=measure(source_on_gpu_input, cy)))
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps([dict(role=r['role'], totalMax=r['total']['maxAbsoluteDifference'],
                       localMax=r['localOnSameGpuInput']['maxAbsoluteDifference'],
                       propagatedMax=r['propagatedSourceInput']['maxAbsoluteDifference']) for r in report['rows']]))
