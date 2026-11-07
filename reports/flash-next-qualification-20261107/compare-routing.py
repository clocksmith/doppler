"""Compare retained, unmodified CPU and GPU routing observations.

Usage: python3 compare-routing.py GPU CPU_TRACE REFERENCE OUTPUT
Diagnostics only; this never changes routing or numerical acceptance.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flash-next-qualification-20261106'))
from verified_tensor_io import read_tensor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(gpu_path, trace_path, reference_path, output):
    gpu_path, trace_path, reference_path, output = map(Path, [gpu_path, trace_path, reference_path, output])
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    gpu_bytes = gpu_path.read_bytes()
    gpu = json.loads(gpu_bytes)
    routing_bytes = Path(str(gpu_path)+'.routing.json').read_bytes()
    routing = json.loads(routing_bytes)
    binding_bytes = (trace_path/'receipt.json').read_bytes()
    binding = json.loads(binding_bytes)
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    require(gpu['completed'] and routing['completed'], 'Incomplete GPU observations')
    require(not gpu['gpuErrors'] and not gpu['cleanupFailures'] and not gpu.get('routingControl'), 'Failed or counterfactual run')
    require(binding['completed'] and binding['samplesAndRouterMetadataMatchReference'], 'Incomplete or divergent CPU trace')
    require(binding['referenceSha256'] == sha(reference_bytes), 'Different reference')
    require(routing['runSha256'] == gpu['runSha256'], 'Different GPU run identities')
    for key in ['manifestSha256', 'inputsSha256']:
        require(gpu[key] == binding[key], f'Different {key}')
    require(gpu['runtimeConfig']['inference']['session']['kvcache']['kvDtype'] == binding['kvStorageDtype'] == 'f32', 'Different KV storage')
    gpu_row, = gpu['rows']
    require(gpu_row['id'] == 'long', 'Expected the bounded long-input observation')
    observations = {r['layer']:r for r in gpu_row['routerObservations']}
    choices = {r['layer']:r for r in routing['rows']}
    require(set(observations) == set(choices) == set(range(48)), 'Expected all 48 router observations')
    rows = []
    for layer in range(48):
        filename = f'router-{layer}.npz'
        file = trace_path/filename
        require(sha(file.read_bytes()) == binding['files'][filename], 'CPU trace hash mismatch')
        cpu = np.load(file, allow_pickle=False)
        source_ids, source_input, source_logits = (cpu[k] for k in ['experts', 'input', 'logits'])
        ids_record = choices[layer]['indices']
        raw = Path(ids_record['file']).read_bytes()
        require(sha(raw) == ids_record['sha256'], 'GPU expert-ID hash mismatch')
        ids = np.frombuffer(raw, '<u4').reshape(ids_record['shape'])
        actual_input = read_tensor(observations[layer]['input'])
        actual_logits = read_tensor(observations[layer]['logits'])
        require(ids.shape == source_ids.shape == (2304, 10), 'Expert selection geometry differs')
        require(actual_input.shape == source_input.shape and actual_logits.shape == source_logits.shape == (2304, 512), 'Router geometry differs')
        require(all(np.isfinite(a).all() for a in [source_input, source_logits]), 'Nonfinite CPU trace')
        differences = np.flatnonzero(np.any(np.sort(ids, axis=1) != np.sort(source_ids, axis=1), axis=1))
        index = int(differences[0]) if len(differences) else None
        delta = actual_input.astype('f8') - source_input.astype('f8')
        rms = np.sqrt(np.mean(source_input.astype('f8')**2, axis=1))
        require(np.all(rms > 0), 'Degenerate source input')
        row = dict(layer=layer, changedSets=len(differences), changedOrder=int(np.sum(np.any(ids != source_ids, axis=1))),
                   inputMaxAbsolute=float(np.abs(delta).max()), inputMaxOverRowRms=float((np.max(np.abs(delta),axis=1)/rms).max()),
                   logitsMaxAbsolute=float(np.abs(actual_logits.astype('f8')-source_logits.astype('f8')).max()))
        if index is not None:
            margin = reference['sequences']['long']['router'][str(layer)]['marginKthToNext'][index]
            relevant = sorted(set(ids[index].tolist()) | set(source_ids[index].tolist()))
            row['first'] = dict(position=index, sourceExperts=source_ids[index].tolist(), gpuExperts=ids[index].tolist(), sourceMargin=margin,
                                scores=[dict(expert=e, source=float(source_logits[index,e]), gpu=float(actual_logits[index,e])) for e in relevant])
        rows.append(row)
    report = dict(scope='Actual all-position routing comparison; no tensor feedback or acceptance-policy change',
                  gpuSha256=sha(gpu_bytes), routingSha256=sha(routing_bytes), sourceTraceSha256=sha(binding_bytes),
                  referenceSha256=sha(reference_bytes), rows=rows,
                  firstChangedLayer=next((r['layer'] for r in rows if r['changedSets']), None),
                  totalChangedSets=sum(r['changedSets'] for r in rows))
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main(*sys.argv[1:])
