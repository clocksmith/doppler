"""Compare observed GPU states against independent native HF at frozen gates."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from binding import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gpu', 'teacher', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gpu_path, teacher_path = args.gpu / 'result.json', args.teacher / 'receipt.json'
    gpu, teacher = json.loads(gpu_path.read_text()), json.loads(teacher_path.read_text())
    identity = json.loads((args.gpu / 'identity.json').read_text())['modelReceipt']
    model_path = Path(identity['path'])
    if digest(model_path) != identity['sha256']:
        raise ValueError('GPU model identity changed')
    policy = json.loads(model_path.read_text())['contract']
    if not gpu['executionCompleted'] or not teacher['completed'] or gpu['canonicalInputs'] != teacher['inputs']:
        raise ValueError('GPU/native completion or canonical identity differs')
    if policy['kvPhysicalDtype'] != 'float32' or teacher['contract']['kvPhysicalDtype'] != 'float32':
        raise ValueError('this observer requires physical F32 for every cache tensor')
    native = {(prompt['id'], snapshot['step'], state['name']): state for prompt in teacher['prompts']
              for snapshot in prompt['snapshots'] for state in snapshot['states']}
    actual = {(state['promptId'], state['step'], state['name']): state for state in gpu['stateObservations']}
    if set(native) != set(actual):
        raise ValueError('GPU/native snapshot inventory differs')
    prompts = {prompt['id']: prompt for prompt in teacher['prompts']}
    generated = {prompt['id']: prompt for prompt in gpu['prompts']}
    receipt = {'schema': 'doppler.canonical-tjs-gpu-state-diagnostic.v1', 'qualification': False,
        'scope': 'Actual GPU-state numerical diagnostic; prior ONNX CPU failures remain retained',
        'sourceSha256': digest(Path(__file__)), 'gpu': {'path': str(gpu_path), 'sha256': digest(gpu_path)},
        'teacher': {'path': str(teacher_path), 'sha256': digest(teacher_path)}, 'model': identity,
        'originalCpuGate': gpu['cpuGate'], 'absoluteTolerance': policy['stateAbsoluteTolerance'],
        'relativeTolerance': policy['stateRelativeTolerance'], 'checks': []}
    for key, state in actual.items():
        target = native[key]
        if state['shape'] != target['shape'] or state['dtype'] != target['dtype']:
            raise ValueError('state physical shape/dtype differs: ' + str(key))
        for root, entry in ((args.gpu, state), (args.teacher, target)):
            path = root / entry['file']
            if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise ValueError('observed state file changed: ' + str(path))
        value = np.fromfile(args.gpu / state['file'], dtype='<f4')
        reference = np.fromfile(args.teacher / target['file'], dtype='<f4')
        error = np.abs(value - reference)
        limits = policy['stateAbsoluteTolerance'] + policy['stateRelativeTolerance'] * np.abs(reference)
        failed = (~np.isfinite(value)) | (~np.isfinite(reference)) | (error > limits)
        prompt_id, step, name = key
        aligned = generated[prompt_id]['generatedTokenIds'][:step] == prompts[prompt_id]['generatedTokenIds'][:step]
        first = int(np.flatnonzero(failed)[0]) if failed.any() else None
        receipt['checks'].append({'promptId': prompt_id, 'step': step, 'name': name, 'shape': state['shape'],
            'prefixAligned': aligned, 'maxAbsoluteError': float(error.max()), 'meanAbsoluteError': float(error.mean()),
            'failedElements': int(failed.sum()), 'pass': aligned and not bool(failed.any()),
            'firstFailedElement': None if first is None else {'index': first, 'actual': float(value[first]),
                'native': float(reference[first]), 'absoluteError': float(error[first]), 'limit': float(limits[first])}})
    receipt['pass'] = bool(receipt['checks']) and all(check['pass'] for check in receipt['checks'])
    receipt['failedChecks'] = sum(not check['pass'] for check in receipt['checks'])
    receipt['failuresByState'] = dict(Counter(check['name'] for check in receipt['checks'] if not check['pass']))
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'checks': len(receipt['checks']), 'failed': receipt['failedChecks'], 'pass': receipt['pass']}))
    if not receipt['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
