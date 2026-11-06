"""Replay layer-0 source GDN on exact captured GPU branch input, CPU only.

Usage: python3 -P check-linear-branch-same-input.py <gpu.json> <producer.py>
       <source-dir> <new-cpu-trace-dir> <artifact> <reference-data> <out.json>
The source helper retains the loaded module, then hooks are removed before
standalone replay so the original observation files cannot be overwritten.
"""
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

gpu_path, producer, source, cpu_dir, artifact, data, output = map(Path, sys.argv[1:])
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes = gpu_path.read_bytes()
gpu = json.loads(gpu_bytes)
assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['attentionProjectionInput']['dtype'] == 'f32'
helper = Path(__file__).with_name('prepare-layer0-upstream-trace.py')
sys.argv = [str(helper), str(producer), str(source), str(cpu_dir), str(artifact), str(data), '[2135]']
scope = runpy.run_path(str(helper), run_name='__main__')
environment = scope['environment']
for hook in environment['hooks']:
    hook.remove()
cpu_bytes = (cpu_dir / 'trace.json').read_bytes()
cpu = json.loads(cpu_bytes)
assert gpu['manifestSha256'] == cpu['manifestSha256'] and gpu['inputsSha256'] == cpu['inputsSha256']
module = environment['layer'].linear_attn
spec = importlib.util.spec_from_file_location('stage_metrics', Path(__file__).with_name('compare-upstream-stages.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

def read_cpu(role, stage):
    record = next(t for t in cpu['tensors'] if t['role'] == role and t['stage'] == stage)
    file = cpu_dir / record['file']
    assert sha(file.read_bytes()) == record['sha256']
    return np.load(file, allow_pickle=False).copy()


def read_gpu(record):
    raw = Path(record['file']).read_bytes()
    assert sha(raw) == record['sha256']
    return np.frombuffer(raw, dtype='<f4').reshape(record['shape']).copy()


source_input = read_cpu('attention', 'projectionInput')
source_output = read_cpu('linear', 'branchOutput')
gpu_input = read_gpu(gpu['attentionProjectionInput'])
record = next(t for t in gpu['tensors'] if t['layer'] == 0 and t['role'] == 'attn_hc' and t['stage'] == 'branchOutput')
assert record['positions'] == [2135]
gpu_output = read_gpu(record)
with torch.no_grad():
    replay = module(torch.from_numpy(source_input).unsqueeze(0)).squeeze(0).numpy().copy()
    assert np.array_equal(replay, source_output), 'fresh-cache source replay must reproduce every original branch output'
    on_gpu_input = module(torch.from_numpy(gpu_input).unsqueeze(0)).squeeze(0).numpy().copy()
report = dict(scope='Unchanged layer-0 CPU GDN on exact GPU input; no tensor feedback or model acceptance',
              normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
              scriptSha256=sha(Path(__file__).read_bytes()), torch=torch.__version__,
              sourceGeometryReplayBitExact=True, layer=0, position=2135,
              input=metrics.error_metrics(gpu_input, source_input),
              total=metrics.error_metrics(gpu_output, source_output[2135:2136]),
              localOnSameGpuInput=metrics.error_metrics(gpu_output, on_gpu_input[2135:2136]),
              propagatedInput=metrics.error_metrics(on_gpu_input[2135:2136], source_output[2135:2136]))
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ['input', 'total', 'localOnSameGpuInput', 'propagatedInput']}, indent=2))
