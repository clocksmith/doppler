"""Compare captured HC means with separate-product and fused CPU controls.

Usage: python3 -P analyze-hc-mixing-order.py <gpu-trace.json> <cpu-trace.json> <out.json> [<layer>]
Readback analysis only. No tensor is fed back into either model.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

assert len(sys.argv) in (4, 5)
gpu_path, cpu_path, output = map(Path, sys.argv[1:4])
gpu_bytes, cpu_bytes = gpu_path.read_bytes(), cpu_path.read_bytes()
gpu, cpu = json.loads(gpu_bytes), json.loads(cpu_bytes)
sha = lambda b: hashlib.sha256(b).hexdigest()
assert gpu['diagnosticCompleted'] and cpu['diagnosticCompleted']
assert not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['manifestSha256'] == cpu['manifestSha256']
assert gpu['inputsSha256'] == cpu['inputsSha256']
layers = sorted(set(t['layer'] for t in gpu['tensors']))
assert len(sys.argv) == 5 or len(layers) == 1, 'multi-layer observations require an explicit layer'
layer = int(sys.argv[4]) if len(sys.argv) == 5 else layers[0]
report = dict(scope='Readback-only HC arithmetic-order controls; no full-model qualification',
              normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
              scriptSha256=sha(Path(__file__).read_bytes()), layer=layer, rows=[])

def read_record(record, source):
    file = Path(record['file'])
    if source == 'cpu':
        file = cpu_path.parent / file
    raw = file.read_bytes()
    assert sha(raw) == record['sha256']
    return np.load(file, allow_pickle=False) if source == 'cpu' else np.frombuffer(raw, dtype='<f4').reshape(record['shape'])

for role in ('attn_hc', 'mlp_hc'):
    records = {stage: next(t for t in gpu['tensors'] if t['layer'] == layer and t['role'] == role and t['stage'] == stage)
               for stage in ('norm', 'mix', 'branchInput')}
    positions = records['norm']['positions']
    assert positions and all(records[stage]['positions'] == positions for stage in records)
    for source, trace in [('gpu', gpu), ('cpu', cpu)]:
        values = {}
        for stage in records:
            record = next(t for t in trace['tensors'] if t['role'] == role and t['stage'] == stage and (source == 'cpu' or t['layer'] == layer))
            values[stage] = read_record(record, source)
            if source == 'cpu':
                if record.get('positions') is not None:
                    assert record['positions'] == positions
                else:
                    values[stage] = values[stage][positions]
        width = values['branchInput'].shape[1]
        norm, mix = (values[key].reshape(len(positions), 4, width) for key in ('norm', 'mix'))
        products = np.multiply(norm, mix, dtype=np.float32)
        separated = (((products[:, 0] + products[:, 1]) + products[:, 2]) + products[:, 3]) * np.float32(0.25)
        mean = torch.from_numpy(products).mean(dim=1).numpy()
        # Float64 intermediate is a diagnostic fused-order pattern, not a new
        # exact-dot or qualification oracle. The physical cancellation fixture
        # independently distinguishes contraction without this approximation.
        fused = np.zeros_like(separated)
        for stream in range(4):
            fused = (norm[:, stream].astype(np.float64) * mix[:, stream].astype(np.float64) + fused).astype(np.float32)
        fused *= np.float32(0.25)
        expected = values['branchInput']
        metrics = {name: dict(maxAbsoluteDifference=float(np.max(np.abs(actual - expected))),
                              differingValues=int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))))
                   for name, actual in [('sourceMean', mean), ('separateProducts', separated), ('fusedPattern', fused)]}
        if source == 'cpu':
            assert metrics['sourceMean']['differingValues'] == 0, 'control must reproduce the pinned source operation'
            assert metrics['separateProducts']['differingValues'] == 0
        report['rows'].append(dict(role=role, source=source, positions=positions, metrics=metrics))
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
