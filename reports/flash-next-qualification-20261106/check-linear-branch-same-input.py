"""Replay layer-0 source GDN on exact captured GPU branch input, CPU only.

Usage: python3 -P check-linear-branch-same-input.py <gpu.json> <producer.py>
       <source-dir> <new-cpu-trace-dir> <artifact> <reference-data> <out.json>
       [--stages]
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

arguments = sys.argv[1:]
with_stages = '--stages' in arguments
if with_stages:
    arguments.remove('--stages')
gpu_path, producer, source, cpu_dir, artifact, data, output = map(Path, arguments)
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
if with_stages:
    mapping = {'qkv': ('in_proj_qkv', 'attn.qkv_proj'), 'z': ('in_proj_z', 'attn.linear_z_proj'),
               'a': ('in_proj_a', 'attn.linear_a_proj'), 'b': ('in_proj_b', 'attn.linear_b_proj')}
    actual = {}
    for key, (_, stage) in {**mapping, 'core': ('norm', 'attn.linear_core_out')}.items():
        record = next(t for t in gpu['tensors'] if t['role'] == 'linear' and t['stage'] == stage)
        actual[key] = read_gpu(record)
        assert actual[key].shape[0] == gpu_input.shape[0]
    input_record = next(t for t in gpu['tensors'] if t['role'] == 'linear' and t['stage'] == 'attn.post_input_norm')
    assert np.array_equal(read_gpu(input_record), gpu_input)

    def replay_stages(replacements):
        observed, handles = {}, []
        def observe(key):
            def hook(_module, _inputs, result):
                if key in replacements:
                    result = torch.from_numpy(replacements[key]).reshape(result.shape)
                observed[key] = result.detach().float().reshape(gpu_input.shape[0], -1).numpy().copy()
                return result if key in replacements else None
            return hook
        for key, (field, _) in mapping.items():
            handles.append(getattr(module, field).register_forward_hook(observe(key)))
        handles.append(module.norm.register_forward_hook(observe('core')))
        try:
            with torch.no_grad():
                result = module(torch.from_numpy(gpu_input).unsqueeze(0)).squeeze(0).numpy().copy()
        finally:
            for handle in handles:
                handle.remove()
        return result, observed

    repeated, on_gpu_stages = replay_stages({})
    assert np.array_equal(repeated, on_gpu_input), 'observation hooks must not change source branch math'
    report['projectionControls'] = []
    for key, (_, stage) in mapping.items():
        report['projectionControls'].append(dict(stage=stage,
            allRows=metrics.error_metrics(actual[key], on_gpu_stages[key]),
            checkedRow=metrics.error_metrics(actual[key][2135], on_gpu_stages[key][2135])))
    report['coreWithSourceProjectionsOnGpuInput'] = metrics.error_metrics(actual['core'][2135], on_gpu_stages['core'][2135])
    _, forced_stages = replay_stages({key: actual[key] for key in mapping})
    report['coreOnExactGpuProjections'] = metrics.error_metrics(actual['core'][2135], forced_stages['core'][2135])
    with torch.no_grad():
        source_output_projection = module.out_proj(torch.from_numpy(actual['core']).unsqueeze(0)).squeeze(0).numpy().copy()
    report['outputProjectionOnExactGpuCore'] = metrics.error_metrics(gpu_output, source_output_projection[2135:2136])
    report['stageControlScope'] = 'CPU module with captured GPU projection outputs, then CPU output projection on exact GPU core; no tensor feedback or model acceptance'
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ['input', 'total', 'localOnSameGpuInput', 'propagatedInput']}, indent=2))
