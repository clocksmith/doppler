"""Pinned-source CPU control: remove only captured V storage rounding.

Usage: python3 check-value-storage-control.py GPU_TRACE CPU_TRACE SOURCE_DATA OUT
No runtime tensor feedback, reference replacement or model qualification.
"""
import __future__
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flash-next-qualification-20261106'))
from verified_tensor_io import read_tensor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(gpu_path, cpu_path, data, output):
    gpu_path, cpu_path, data, output = map(Path, [gpu_path, cpu_path, data, output])
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    gpu_bytes, cpu_bytes = gpu_path.read_bytes(), cpu_path.read_bytes()
    gpu, cpu = json.loads(gpu_bytes), json.loads(cpu_bytes)
    require(gpu['diagnosticCompleted'] and cpu['diagnosticCompleted'], 'Incomplete captures')
    require(not gpu['gpuErrors'] and not gpu['cleanupFailures'], 'Failed GPU capture')
    for key in ['manifestSha256', 'inputsSha256']:
        require(gpu[key] == cpu[key], f'Unmatched {key}')
    kv = gpu['attentionKv']
    require(kv['dtype'] == 'f16' and kv['layout'] == 'contiguous', 'Expected the F16 cache control')
    tokens, heads, kv_heads, dim = (kv[k] for k in ['tokens', 'numHeads', 'numKVHeads', 'headDim'])
    position = 2135
    require(position < tokens and heads % kv_heads == 0, 'Invalid frozen query geometry')
    binding = json.loads((data / 'source-binding.json').read_bytes())
    source = (data / 'source/modeling_qwen4_exp.py').read_bytes()
    pinned = next(r for r in binding['referenceImplementation']['files'] if r['path'] == 'modeling_qwen4_exp.py')
    require(sha(source) == pinned['sha256'], 'Source implementation drift')
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in ['repeat_kv', 'eager_attention_forward']]
    require(len(nodes) == 2 and not any(n.decorator_list for n in nodes), 'Unexpected source functions')
    namespace = dict(torch=torch, nn=torch.nn, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned-attention', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    torch.set_num_threads(16)

    def array(trace, parent, stage):
        record = next(r for r in trace['tensors'] if r['role'] == 'attention' and r['stage'] == stage)
        if stage == 'mask':
            # Additive masks legitimately contain -inf; other tensors retain the
            # shared reader's strict finite-value requirement.
            file = Path(record['file'])
            raw = (file if file.is_absolute() else parent / file).read_bytes()
            require(sha(raw) == record['sha256'], 'Mask hash mismatch')
            values = np.load(io.BytesIO(raw), allow_pickle=False)
            require(values.dtype == np.dtype('float32') and list(values.shape) == record['shape'], 'Mask shape or dtype mismatch')
            require(np.all((values == 0) | (values == np.finfo(np.float32).min) | np.isneginf(values)),
                    'Expected zero, minimum-F32 or negative-infinity additive mask entries')
            return values.copy()
        return read_tensor(record, parent).copy()

    def cached(kind):
        record = next(r for r in kv['tensors'] if r['kind'] == kind)
        raw = Path(record['file']).read_bytes()
        require(sha(raw) == record['sha256'], 'Cache bytes do not match their receipt')
        require(len(raw) == tokens * kv_heads * dim * 2, 'Wrong cache extent')
        values = np.frombuffer(raw, '<f2').astype(np.float32).reshape(tokens, kv_heads * dim)
        require(np.isfinite(values).all(), 'Nonfinite cache')
        return values

    source_values = [array(cpu, cpu_path.parent, name) for name in ['query', 'cachedKey', 'cachedValue']]
    gpu_values = [array(gpu, gpu_path.parent, 'attn.q_rope'), cached('key'), cached('value')]
    gpu_value_f32 = array(gpu, gpu_path.parent, 'attn.v_proj')
    source_value_f32 = array(cpu, cpu_path.parent, 'attn.v_proj')
    require(np.array_equal(gpu_value_f32.astype('<f2').astype(np.float32), gpu_values[2]), 'GPU cache is not the projected F16 value')
    require(np.array_equal(source_value_f32.astype('<f2').astype(np.float32), source_values[2]), 'Source cache control is not rounded V')
    mask_row = array(cpu, cpu_path.parent, 'mask').reshape(tokens)
    selection = next(r for r in gpu['qsaObservations'] if r['layer'] == kv['layer'])
    selected = next(r for r in selection['rows'] if r['position'] == position)
    require(selected['indices'] == np.flatnonzero(mask_row == 0).tolist(), 'Different sparse selection')
    mask = torch.zeros((1, 1, tokens, tokens), dtype=torch.float32)
    mask[0, 0, position].copy_(torch.from_numpy(mask_row))
    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, training=False)

    def evaluate(values):
        shaped = [torch.from_numpy(v.reshape(1, tokens, count, dim)).transpose(1, 2)
                  for v, count in zip(values, [heads, kv_heads, kv_heads])]
        with torch.no_grad():
            result, _ = namespace['eager_attention_forward'](module, *shaped, mask,
                                                            scaling=dim ** -0.5, dropout=0.0)
        return result.reshape(tokens, heads * dim)[position:position+1].numpy().copy()

    original = array(cpu, cpu_path.parent, 'attn.core_out')
    require(np.array_equal(evaluate(source_values), original), 'Original CPU geometry does not replay bit-exactly')

    def metrics(values, reference):
        delta = values.astype('f8')-reference.astype('f8')
        return dict(maxAbsolute=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta*delta))))

    actual = array(gpu, gpu_path.parent, 'attn.core_out')[position:position+1]
    candidate = [*gpu_values[:2], gpu_value_f32]
    f32_source = [*source_values[:2], source_value_f32]
    candidate_out, f32_source_out = evaluate(candidate), evaluate(f32_source)
    report = dict(scope='CPU-only V storage control at one frozen query; not a full-model candidate or acceptance',
                  normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
                  sourceSha256=sha(source), scriptSha256=sha(Path(__file__).read_bytes()),
                  layer=kv['layer'], position=position, tokens=tokens, torch=torch.__version__,
                  originalSourceReplayBitExact=True, baselineGpu=metrics(actual, original),
                  sourceOnGpuF16Inputs=metrics(evaluate(gpu_values), original),
                  gpuF32ValueAgainstExistingF16Reference=metrics(candidate_out, original),
                  sourceF32ValueAgainstExistingF16Reference=metrics(f32_source_out, original),
                  gpuF32ValueAgainstMatchedF32Source=metrics(candidate_out, f32_source_out))
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
