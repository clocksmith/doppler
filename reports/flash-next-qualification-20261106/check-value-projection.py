"""Separate value-projection drift from F16 cache encoding on captured inputs.

Usage: python3 -P check-value-projection.py <gpu.json> <cpu-trace.json> <artifact>
       <reference-data> <pinned-producer.py> <out.json>
Readback-only CPU controls; never a model acceptance path or tensor feedback.
"""
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gguf import quants, GGMLQuantizationType

gpu_path, cpu_path, artifact, data, producer, output = map(Path, sys.argv[1:])
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes, cpu_bytes, manifest_bytes = gpu_path.read_bytes(), cpu_path.read_bytes(), (artifact / 'manifest.json').read_bytes()
gpu, cpu, manifest = json.loads(gpu_bytes), json.loads(cpu_bytes), json.loads(manifest_bytes)
assert gpu['diagnosticCompleted'] and cpu['diagnosticCompleted']
assert not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['manifestSha256'] == cpu['manifestSha256'] == sha(manifest_bytes)
assert gpu['inputsSha256'] == cpu['inputsSha256']
assert sha(producer.read_bytes()) == cpu['producerSha256']
kv = gpu['attentionKv']
layer, tokens, width = kv['layer'], kv['tokens'], manifest['architecture']['hiddenSize']
kv_width = kv['numKVHeads'] * kv['headDim']
name = f'model.language_model.layers.{layer}.self_attn.v_proj.weight'
headers = json.loads((data / 'checkpoint-headers.json').read_bytes())
locations = {name: (file, info['dataStart'], tensor) for file, info in headers['files'].items() for name, tensor in info['tensors'].items()}
assert locations[name][2]['dtype'] != 'I64'
node = next(n for n in ast.parse(producer.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'read_tensor')
namespace = dict(artifact_manifest=manifest, ARTIFACT=artifact, locations=locations, torch=torch,
                 np=np, math=math, quants=quants, GGMLQuantizationType=GGMLQuantizationType, DTYPE=torch.float32)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned-artifact-weight-reader', 'exec'), namespace)
weights = namespace['read_tensor'](name)
assert weights.shape == (kv_width, width)
spec = importlib.util.spec_from_file_location('stage_metrics', Path(__file__).with_name('compare-upstream-stages.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
torch.set_num_threads(16)

def read_cpu(stage):
    record = next(t for t in cpu['tensors'] if t['role'] == 'attention' and t['stage'] == stage)
    file = cpu_path.parent / record['file']
    assert sha(file.read_bytes()) == record['sha256']
    return np.load(file, allow_pickle=False).copy()


def read_gpu(stage, columns):
    record = gpu['attentionProjectionInput'] if stage == 'projectionInput' else next(
        t for t in gpu['tensors'] if t['role'] == 'attention' and t['stage'] == stage)
    raw = Path(record['file']).read_bytes()
    assert sha(raw) == record['sha256']
    return np.frombuffer(raw, dtype='<f4').reshape(tokens, columns).copy()


source_input, source_value = read_cpu('projectionInput'), read_cpu('attn.v_proj')
gpu_input, gpu_value = read_gpu('projectionInput', width), read_gpu('attn.v_proj', kv_width)
with torch.no_grad():
    replay = F.linear(torch.from_numpy(source_input), weights).numpy()
    assert np.array_equal(replay, source_value), 'same-shape CPU replay must reproduce every source V projection value'
    source_on_gpu_input = F.linear(torch.from_numpy(gpu_input), weights).numpy()
cache_record = next(t for t in kv['tensors'] if t['kind'] == 'value')
cache_bytes = Path(cache_record['file']).read_bytes()
assert sha(cache_bytes) == cache_record['sha256']
cache = np.frombuffer(cache_bytes, dtype='<f2').reshape(tokens, kv_width)
rounded_gpu = gpu_value.astype('<f2')
cache_mismatch = cache.view('<u2') != rounded_gpu.view('<u2')
source_cache = read_cpu('cachedValue').astype('<f2')
assert np.array_equal(source_cache, source_value.astype('<f2'))
report = dict(scope='Learned V projection and exact F16 cache encoding controls; not model acceptance', normalModelQualified=False,
              gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes), manifestSha256=sha(manifest_bytes),
              producerSha256=sha(producer.read_bytes()), scriptSha256=sha(Path(__file__).read_bytes()),
              weightsF32Sha256=sha(weights.numpy().tobytes()), torch=torch.__version__, threads=torch.get_num_threads(), layer=layer,
              sourceGeometryReplayBitExact=True, comparedValues=int(cache.size),
              cacheBitsDifferFromCpuRounding=int(np.count_nonzero(cache_mismatch)),
              cacheBitsDifferFromSource=int(np.count_nonzero(cache.view('<u2') != source_cache.view('<u2'))),
              totalProjection=metrics.error_metrics(gpu_value, source_value),
              localProjectionOnSameGpuInput=metrics.error_metrics(gpu_value, source_on_gpu_input),
              propagatedInput=metrics.error_metrics(source_on_gpu_input, source_value),
              input=metrics.error_metrics(gpu_input, source_input),
              mismatchedCacheExamples=[dict(row=int(row), column=int(column), gpu=float(cache[row, column]),
                                            cpuRounded=float(rounded_gpu[row, column]), projected=float(gpu_value[row, column]))
                                      for row, column in np.argwhere(cache_mismatch)[:8]])
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ['comparedValues', 'cacheBitsDifferFromCpuRounding', 'cacheBitsDifferFromSource',
                                       'totalProjection', 'localProjectionOnSameGpuInput', 'propagatedInput']}, indent=2))
