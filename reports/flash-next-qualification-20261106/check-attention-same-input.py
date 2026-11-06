"""Replay the pinned CPU attention on exact captured GPU Q and F16 KV bytes.

Usage: python3 -P check-attention-same-input.py <gpu.json> <cpu-trace.json>
       <reference-data> <position> <out.json>
Operator-only readback analysis, never tensor feedback or model acceptance.
"""
import __future__
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

gpu_path, cpu_path, data = map(Path, sys.argv[1:4])
position, output = int(sys.argv[4]), Path(sys.argv[5])
sha = lambda b: hashlib.sha256(b).hexdigest()
gpu_bytes, cpu_bytes = gpu_path.read_bytes(), cpu_path.read_bytes()
gpu, cpu = json.loads(gpu_bytes), json.loads(cpu_bytes)
assert gpu['diagnosticCompleted'] and cpu['diagnosticCompleted']
assert not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['manifestSha256'] == cpu['manifestSha256'] and gpu['inputsSha256'] == cpu['inputsSha256']
kv = gpu['attentionKv']
assert kv['layout'] == 'contiguous' and kv['dtype'] == 'f16'
tokens, heads, kv_heads, dim = kv['tokens'], kv['numHeads'], kv['numKVHeads'], kv['headDim']
assert heads % kv_heads == 0 and 0 <= position < tokens
binding = json.loads((data / 'source-binding.json').read_bytes())
source = (data / 'source/modeling_qwen4_exp.py').read_bytes()
entry = next(r for r in binding['referenceImplementation']['files'] if r['path'] == 'modeling_qwen4_exp.py')
assert sha(source) == entry['sha256']
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in ('repeat_kv', 'eager_attention_forward')]
assert len(nodes) == 2 and all(not n.decorator_list for n in nodes)
namespace = dict(torch=torch, nn=torch.nn, F=F)
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned-eager-attention', 'exec',
             flags=__future__.annotations.compiler_flag), namespace)
spec = importlib.util.spec_from_file_location('stage_metrics', Path(__file__).with_name('compare-upstream-stages.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
torch.set_num_threads(16)

def cpu_array(stage):
    record = next(r for r in cpu['tensors'] if r['role'] == 'attention' and r['stage'] == stage)
    file = cpu_path.parent / record['file']
    assert sha(file.read_bytes()) == record['sha256']
    return np.load(file, allow_pickle=False).copy()


def gpu_array(stage):
    record = next(r for r in gpu['tensors'] if r['role'] == 'attention' and r['stage'] == stage)
    raw = Path(record['file']).read_bytes()
    assert sha(raw) == record['sha256']
    return np.frombuffer(raw, dtype='<f4').copy()


def shaped(values, count):
    return torch.from_numpy(values.reshape(1, tokens, count, dim)).transpose(1, 2)


source_values = [cpu_array(stage) for stage in ('query', 'cachedKey', 'cachedValue')]
gpu_values = [gpu_array('attn.q_rope').reshape(tokens, heads * dim)]
for kind in ('key', 'value'):
    record = next(r for r in kv['tensors'] if r['kind'] == kind)
    raw = Path(record['file']).read_bytes()
    assert sha(raw) == record['sha256']
    gpu_values.append(np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(tokens, kv_heads * dim))
mask_row = cpu_array('mask').reshape(tokens)
layer = next(r for r in gpu['qsaObservations'] if r['layer'] == kv['layer'])
selection = next(r for r in layer['rows'] if r['position'] == position)
assert selection['indices'] == np.flatnonzero(mask_row == 0).tolist(), 'CPU mask must select the same actual GPU token rows'
mask = torch.zeros((1, 1, tokens, tokens), dtype=torch.float32)
mask[0, 0, position].copy_(torch.from_numpy(mask_row))
module = SimpleNamespace(num_key_value_groups=heads // kv_heads, training=False)
source_output = cpu_array('attn.core_out')
assert source_output.shape == (1, heads * dim)
gpu_output = gpu_array('attn.core_out').reshape(tokens, heads * dim)[position:position + 1]

def evaluate(values):
    query, key, value = (shaped(x, count) for x, count in zip(values, (heads, kv_heads, kv_heads)))
    with torch.no_grad():
        result, _ = namespace['eager_attention_forward'](module, query, key, value, mask,
                                                       scaling=dim ** -0.5, dropout=0.0)
    return result.reshape(tokens, heads * dim)[position:position + 1].numpy().copy()


baseline = evaluate(source_values)
assert np.array_equal(baseline, source_output), 'full-geometry CPU replay must reproduce the frozen original core output exactly'
on_gpu_input = evaluate(gpu_values)
report = dict(scope='Pinned CPU attention on actual GPU inputs; diagnostic only, no tensor feedback or gate replacement',
              normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
              sourceSha256=sha(source), scriptSha256=sha(Path(__file__).read_bytes()),
              layer=kv['layer'], position=position, torch=torch.__version__, threads=torch.get_num_threads(),
              sourceGeometryReplayBitExact=True,
              total=metrics.error_metrics(gpu_output, source_output),
              localOnSameGpuInputs=metrics.error_metrics(gpu_output, on_gpu_input),
              propagatedInputs=metrics.error_metrics(on_gpu_input, source_output), inputs=[], singleInputControls=[])
for index, name in enumerate(('query', 'key', 'value')):
    visible_source, visible_gpu = source_values[index][:position + 1], gpu_values[index][:position + 1]
    report['inputs'].append(dict(name=name, differingValues=int(np.count_nonzero(visible_source != visible_gpu)),
                                 **metrics.error_metrics(visible_gpu, visible_source)))
    values = list(source_values)
    values[index] = gpu_values[index]
    result = evaluate(values)
    report['singleInputControls'].append(dict(input=name, **metrics.error_metrics(result, source_output)))
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ('layer', 'position', 'total', 'localOnSameGpuInputs', 'propagatedInputs')}, indent=2))
