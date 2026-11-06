"""Estimate each layer's attention-branch output error at token 2135 from the
retained relative-prefix GPU slices and frozen CPU source slices, and split the
injected residual error into gate-scaled branch error versus gate error.

Neither capture retained the branch output itself, so both sides recover it from
the strongest stream as (mlp_hc.input - attn_hc.input) / gate, which the GPU
slices confirm as the injection law to F32 rounding. Readback analysis only; no
tensor feedback and no model acceptance claim.

Usage: python3 -P estimate-injection-steps.py <qualify-dir> <out.json>
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
G = Q / 'relative-prefix-gpu-20261106.json'
POSITION, STREAMS, WIDTH = 2135, 4, 2560
gpu_trace, gpu_index, gpu_verification = load_verified_trace(G, ('layer', 'role', 'stage'))


def gpu(layer, role, stage):
    record = gpu_index[(layer, role, stage)]
    if record.get('positions') != [POSITION]:
        raise ValueError('Injection estimates require the recorded token position')
    return read_tensor(record, Q).reshape(-1).astype(np.float64)


def cpu(layer, role, stage):
    base = Q / f'relative-source-L{layer}-20261106'
    trace = json.loads((base / 'trace.json').read_bytes())
    record = next(t for t in trace['tensors'] if t['role'] == role and t['stage'] == stage)
    values = read_tensor(record, base)
    if record['positions'] is None:
        values = values[POSITION]
    return values.reshape(-1).astype(np.float64)


def streams(vector):
    return vector.reshape(STREAMS, WIDTH)


rows = []
for layer in range(16):
    g_in, g_gate, g_inj = gpu(layer, 'attn_hc', 'input'), gpu(layer, 'attn_hc', 'gate'), gpu(layer, 'mlp_hc', 'input')
    c_in, c_gate, c_inj = cpu(layer, 'attn_hc', 'input'), cpu(layer, 'attn_hc', 'gate'), cpu(layer, 'mlp_hc', 'input')
    strongest = int(np.argmax(np.abs(c_gate)))
    g_out = (streams(g_inj)[strongest] - streams(g_in)[strongest]) / g_gate[strongest]
    c_out = (streams(c_inj)[strongest] - streams(c_in)[strongest]) / c_gate[strongest]
    d_in, d_inj, d_out = streams(g_in - c_in), streams(g_inj - c_inj), g_out - c_out
    law = streams(g_inj) - streams(g_in) - g_gate[:, None] * g_out[None, :]
    out_rms = float(np.sqrt(np.mean(c_out ** 2)))
    per_stream = []
    for s in range(STREAMS):
        per_stream.append(dict(stream=s, gateCpu=float(c_gate[s]), gateGpu=float(g_gate[s]),
                               inputMaxAbsError=float(np.max(np.abs(d_in[s]))),
                               injectedMaxAbsError=float(np.max(np.abs(d_inj[s]))),
                               injectionDeltaMaxAbs=float(np.max(np.abs(d_inj[s] - d_in[s]))),
                               gateTimesOutErrorMaxAbs=float(np.max(np.abs(c_gate[s] * d_out))),
                               gateErrorTimesOutMaxAbs=float(np.max(np.abs((g_gate[s] - c_gate[s]) * g_out))),
                               injectionLawResidualGpuMaxAbs=float(np.max(np.abs(law[s])))))
    rows.append(dict(layer=layer, estimatedFromStream=strongest, branchOutputRms=out_rms,
                     branchOutputMaxAbs=float(np.max(np.abs(c_out))),
                     branchOutputMaxAbsError=float(np.max(np.abs(d_out))),
                     branchOutputRmsError=float(np.sqrt(np.mean(d_out ** 2))),
                     branchOutputMaxErrorOverRms=float(np.max(np.abs(d_out)) / out_rms),
                     branchOutputRmsErrorOverRms=float(np.sqrt(np.mean(d_out ** 2)) / out_rms),
                     streams=per_stream))
OUT.write_text(json.dumps(dict(scope='Estimated attention-branch output error and injected-error split at token 2135; diagnostic only',
                               normalModelQualified=False, position=POSITION, gpuTensorVerification=gpu_verification,
                               scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               tensorReaderSha256=hashlib.sha256(Path(__file__).with_name('verified_tensor_io.py').read_bytes()).hexdigest(),
                               rows=rows), indent=1, allow_nan=False) + '\n')
for r in rows:
    print('L%2d attn branch (est. stream %d): rms=%.4f maxAbs=%.3f maxErr=%.3e rmsErr=%.3e maxErr/rms=%.3e rmsErr/rms=%.3e' % (
        r['layer'], r['estimatedFromStream'], r['branchOutputRms'], r['branchOutputMaxAbs'], r['branchOutputMaxAbsError'],
        r['branchOutputRmsError'], r['branchOutputMaxErrorOverRms'], r['branchOutputRmsErrorOverRms']))
