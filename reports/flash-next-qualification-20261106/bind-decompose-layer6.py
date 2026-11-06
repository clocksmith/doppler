"""Bind the layer-6 staged capture to the earlier normal captures, check CPU
trace determinism, and decompose the token-2135 injection steps with exact GPU
branch outputs for layers 0-6 (attention) and 1-5 (MoE). CPU-only readback
analysis of retained files; no tensor feedback, no model acceptance claim.

Usage: python3 -P bind-and-decompose-layer6.py <qualify-dir> <out.json>
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
tensor_io = runpy.run_path(str(Path(__file__).with_name('verified_tensor_io.py')))
load_verified_trace, read_tensor = tensor_io['load_verified_trace'], tensor_io['read_tensor']

Q = Path(sys.argv[1])
OUT = Path(sys.argv[2])
POSITION, STREAMS, WIDTH = 2135, 4, 2560
STAGED = Q / 'linear-layer6-stages-20261106.json'
RELATIVE = Q / 'relative-prefix-gpu-20261106.json'
LAYER0 = Q / 'linear-layer0-stages-20261106.json'
sha = lambda b: hashlib.sha256(b).hexdigest()
staged, staged_index, staged_verification = load_verified_trace(STAGED, ('layer', 'role', 'stage'))
assert staged['diagnosticCompleted'] and not staged['gpuErrors'] and not staged['cleanupFailures']
report = dict(scope='Binding, determinism and exact injection decomposition for the layer-6 staged capture; diagnostic only',
              normalModelQualified=False, stagedTraceSha256=sha(STAGED.read_bytes()), position=POSITION,
              scriptSha256=sha(Path(__file__).read_bytes()),
              tensorReaderSha256=sha(Path(__file__).with_name('verified_tensor_io.py').read_bytes()))
report['tensorVerification'] = {'staged': staged_verification, 'comparedTraces': []}

# 1. Every hyper-connection slice of layers 0-6 against the normal relative-prefix capture and the layer-0 staged capture.
def bind(other_path, label):
    other, verified, proof = load_verified_trace(other_path, ('layer', 'role', 'stage'))
    report['tensorVerification']['comparedTraces'].append(proof)
    for field in ('manifestSha256', 'inputsSha256'):
        if other[field] != staged[field]:
            raise ValueError(f'Trace identity differs: {field}')
    index = verified
    rows = dict(label=label, traceSha256=sha(other_path.read_bytes()), bitExact=0, different=[], notInOther=0)
    for t in staged['tensors']:
        if t['role'] not in ('attn_hc', 'mlp_hc'):
            continue
        key = (t['layer'], t['role'], t['stage'])
        if key not in index:
            rows['notInOther'] += 1
        elif all(index[key].get(field) == t.get(field) for field in ('sha256', 'shape', 'positions')):
            rows['bitExact'] += 1
        else:
            rows['different'].append(list(key))
    return rows
report['bindings'] = [bind(RELATIVE, 'normal relative-prefix capture, layers 0-15'), bind(LAYER0, 'layer-0 staged capture')]

# 2. CPU trace determinism: the standalone stage-source run against the in-check run.
def trace_files(directory):
    trace, _, proof = load_verified_trace(directory / 'trace.json', ('role', 'stage'))
    report['tensorVerification']['comparedTraces'].append(proof)
    return {(t['role'], t['stage']): t['sha256'] for t in trace['tensors']}, trace
source_files, source_trace = trace_files(Q / 'linear-layer6-stage-source-20261106')
check_files, check_trace = trace_files(Q / 'linear-layer6-stage-check-20261106')
for field in ('manifestSha256', 'inputsSha256', 'parentReceiptSha256', 'checkpointSha256'):
    if source_trace[field] != check_trace[field]:
        raise ValueError(f'CPU replay identity differs: {field}')
report['cpuTraceDeterminism'] = dict(tensors=len(source_files), identical=sum(1 for k in source_files if check_files.get(k) == source_files[k]),
                                    differing=[list(k) for k in source_files if check_files.get(k) != source_files[k]],
                                    parentReceiptSha256=source_trace.get('parentReceiptSha256'), checkpointSha256=source_trace.get('checkpointSha256'))

# 3. Exact decomposition helpers.
def gpu(layer, role, stage):
    record = staged_index[(layer, role, stage)]
    if record.get('positions') != [POSITION]:
        raise ValueError('Injection decomposition requires the recorded token position')
    return read_tensor(record, Q).reshape(-1).astype(np.float64)


def cpu(layer, role, stage):
    base = Q / f'relative-source-L{layer}-20261106'
    trace = json.loads((base / 'trace.json').read_bytes())
    record = next((t for t in trace['tensors'] if t['role'] == role and t['stage'] == stage), None)
    if record is None:
        return None
    values = read_tensor(record, base)
    if record['positions'] is None:
        values = values[POSITION]
    return values.reshape(-1).astype(np.float64)


def cpu_layer6(role, stage):
    base = Q / 'linear-layer6-stage-source-20261106'
    record = next(t for t in source_trace['tensors'] if t['role'] == role and t['stage'] == stage)
    values = read_tensor(record, base)
    if record['positions'] is None:
        values = values[POSITION]
    return values.reshape(-1).astype(np.float64)


def metrics(actual, reference):
    diff = actual - reference
    rms = float(np.sqrt(np.mean(reference ** 2)))
    return dict(referenceRms=rms, referenceMaxAbsolute=float(np.max(np.abs(reference))),
                maxAbsoluteDifference=float(np.max(np.abs(diff))), rmsDifference=float(np.sqrt(np.mean(diff ** 2))),
                maxErrorOverReferenceRms=float(np.max(np.abs(diff)) / rms) if rms else None,
                rmsErrorOverReferenceRms=float(np.sqrt(np.mean(diff ** 2)) / rms) if rms else None,
                argmaxDifference=int(np.argmax(np.abs(diff))))


def streams(v):
    return v.reshape(STREAMS, WIDTH)


def injection(layer, role, g_in, g_out, g_gate, g_inj, c_in, c_out, c_gate, c_inj, out_source):
    law = streams(g_inj) - streams(g_in) - g_gate[:, None] * g_out[None, :]
    d_out = g_out - c_out
    per_stream = []
    for s in range(STREAMS):
        delta = (streams(g_inj)[s] - streams(c_inj)[s]) - (streams(g_in)[s] - streams(c_in)[s])
        j = int(np.argmax(np.abs(delta)))
        per_stream.append(dict(stream=s, gateCpu=float(c_gate[s]), gateGpu=float(g_gate[s]), gateDifference=float(g_gate[s] - c_gate[s]),
                               inputMaxAbsError=float(np.max(np.abs(streams(g_in)[s] - streams(c_in)[s]))),
                               injectedMaxAbsError=float(np.max(np.abs(streams(g_inj)[s] - streams(c_inj)[s]))),
                               injectionDeltaMaxAbs=float(abs(delta[j])), atElement=j, branchOutputThere=float(c_out[j]),
                               gateTimesOutErrorThere=float(c_gate[s] * d_out[j]), gateErrorTimesOutThere=float((g_gate[s] - c_gate[s]) * g_out[j]),
                               injectionLawResidualMaxAbs=float(np.max(np.abs(law[s])))))
    return dict(layer=layer, role=role, branchOutputSource=out_source, branchOutput=metrics(g_out, c_out), streams=per_stream)


rows = []
for layer in range(7):
    # Attention branch: exact GPU output; CPU output exact at layer 6, estimated elsewhere from the strongest stream.
    g_in, g_out, g_gate, g_inj = (gpu(layer, 'attn_hc', s) for s in ('input', 'branchOutput', 'gate', 'injected'))
    if layer == 6:
        c_in, c_out, c_gate, c_inj = (cpu_layer6('attn_hc', s) for s in ('input', 'branchOutput', 'gate', 'injected'))
        source = 'cpu-trace'
    else:
        c_in, c_gate, c_inj = cpu(layer, 'attn_hc', 'input'), cpu(layer, 'attn_hc', 'gate'), cpu(layer, 'mlp_hc', 'input')
        s = int(np.argmax(np.abs(c_gate)))
        c_out = (streams(c_inj)[s] - streams(c_in)[s]) / c_gate[s]
        source = f'estimated from CPU stream {s}: (mlp_hc.input - attn_hc.input) / gate'
    rows.append(injection(layer, 'attn_hc', g_in, g_out, g_gate, g_inj, c_in, c_out, c_gate, c_inj, source))
    if layer == 6:
        continue
    # MoE branch of layers 1-5: exact GPU output; CPU output estimated from the next layer's attention input.
    g_in, g_out, g_gate, g_inj = (gpu(layer, 'mlp_hc', s) for s in ('input', 'branchOutput', 'gate', 'injected'))
    if layer == 0:
        rows.append(dict(layer=0, role='mlp_hc', branchOutputSource='not estimated: the n-gram PLE layer runs between the layer-0 injection and the layer-1 attention input, and no CPU slice of the layer-0 MoE output was retained',
                         gpuInjectedEqualsNextLayerInput=bool(np.array_equal(g_inj, gpu(1, 'attn_hc', 'input')))))
        continue
    c_in, c_gate = cpu(layer, 'mlp_hc', 'input'), cpu(layer, 'mlp_hc', 'gate')
    c_inj = cpu_layer6('attn_hc', 'input') if layer + 1 == 6 else cpu(layer + 1, 'attn_hc', 'input')
    next_gpu_input = gpu(layer + 1, 'attn_hc', 'input')
    s = int(np.argmax(np.abs(c_gate)))
    c_out = (streams(c_inj)[s] - streams(c_in)[s]) / c_gate[s]
    row = injection(layer, 'mlp_hc', g_in, g_out, g_gate, g_inj, c_in, c_out, c_gate, c_inj,
                    f'estimated from CPU stream {s}: (next attn_hc.input - mlp_hc.input) / gate')
    row['gpuInjectedEqualsNextLayerInput'] = bool(np.array_equal(g_inj, next_gpu_input))
    rows.append(row)
report['rows'] = rows
OUT.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
for b in report['bindings']:
    print(b['label'], '| bit-exact', b['bitExact'], '| different', len(b['different']), b['different'][:6], '| not in other', b['notInOther'])
d = report['cpuTraceDeterminism']
print('cpu determinism: identical %d/%d differing %s' % (d['identical'], d['tensors'], d['differing']))
for r in rows:
    if 'branchOutput' not in r:
        print('L%d %-7s %s' % (r['layer'], r['role'], r['branchOutputSource']))
        continue
    m = r['branchOutput']
    print('L%d %-7s out[%s] refRMS=%.4f refMax=%.3f maxErr=%.3e rmsErr=%.3e max/RMS=%.3e rms/RMS=%.3e%s' % (
        r['layer'], r['role'], r['branchOutputSource'][:9], m['referenceRms'], m['referenceMaxAbsolute'], m['maxAbsoluteDifference'],
        m['rmsDifference'], m['maxErrorOverReferenceRms'], m['rmsErrorOverReferenceRms'],
        '' if 'gpuInjectedEqualsNextLayerInput' not in r else ' | injected==next input: %s' % r['gpuInjectedEqualsNextLayerInput']))
    for s in r['streams']:
        print('   s%d gate=%.5f dGate=%+.2e inErr=%.2e injErr=%.2e delta=%.2e @%d out=%+.3f gate*dOut=%+.2e dGate*out=%+.2e law=%.1e' % (
            s['stream'], s['gateCpu'], s['gateDifference'], s['inputMaxAbsError'], s['injectedMaxAbsError'], s['injectionDeltaMaxAbs'],
            s['atElement'], s['branchOutputThere'], s['gateTimesOutErrorThere'], s['gateErrorTimesOutThere'], s['injectionLawResidualMaxAbs']))
