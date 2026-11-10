"""Trace captured router inputs after repair; these are not residual-stream states.

Usage: trace-repaired-inputs.py PROTOCOL CPU_ROUTING_TRACE OBSERVATION NEW_OUTPUT
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

protocol_path, cpu_root, observation_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
p = json.loads(protocol_path.read_bytes())
receipt_raw = (cpu_root / 'receipt.json').read_bytes()
receipt = json.loads(receipt_raw)
assert sha(receipt_raw) == p['bindings']['cpuRoutingTraceReceiptSha256']
raw = observation_path.read_bytes()
d = json.loads(raw)
assert d['completed'] and not d['gpuErrors'] and not d['cleanupFailures']
assert d['manifestSha256'] == p['bindings']['manifestSha256']
assert d['inputsSha256'] == p['bindings']['inputsSha256']
records = d['rows'][0]['routerObservations']
assert [r['layer'] for r in records] == list(range(48))
rows = []
for record in records[:26]:
    layer = record['layer']
    raw_input = Path(record['input']['file']).read_bytes()
    assert sha(raw_input) == record['input']['sha256']
    actual = np.frombuffer(raw_input, '<f4').reshape(2304, 2560).astype('f8')
    name = f'router-{layer}.npz'
    assert sha((cpu_root / name).read_bytes()) == receipt['files'][name]
    with np.load(cpu_root / name, allow_pickle=False) as data:
        source = data['input'].astype('f8')
    assert source.shape == actual.shape and np.isfinite(actual).all() and np.isfinite(source).all()
    delta = actual - source
    def metrics(a, r, e):
        source_rms = float(np.sqrt(np.mean(r * r)))
        return dict(sourceRms=source_rms, maxAbsolute=float(np.max(np.abs(e))),
                    differenceRms=float(np.sqrt(np.mean(e * e))),
                    relativeRms=float(np.sqrt(np.sum(e * e) / np.sum(r * r))))
    rows.append(dict(layer=layer, inputSha256=record['input']['sha256'],
                     values=actual.size, allPositions=metrics(actual, source, delta),
                     positions=[dict(position=i, **metrics(actual[i], source[i], delta[i])) for i in p['positions']]))
result = dict(scope='Post-repair captured MLP-router branch inputs at matched boundaries; not full residual-stream captures or operator attribution',
              normalModelQualified=False, acceptanceChanged=False, newGpuInvocations=0,
              observationSha256=sha(raw), cpuRoutingTraceReceiptSha256=sha(receipt_raw),
              producerSha256=sha(Path(__file__).read_bytes()), rows=rows)
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([dict(layer=r['layer'], relativeRms=r['allPositions']['relativeRms'],
                       at1717=r['positions'][0]['relativeRms']) for r in rows], indent=2))
