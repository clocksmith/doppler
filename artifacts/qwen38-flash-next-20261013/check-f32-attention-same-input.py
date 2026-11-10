"""Replay the unchanged F32 attention branch on a captured full GPU input.

Usage: python3 -P check-f32-attention-same-input.py GPU_BOUNDARY PRODUCER SOURCE
       NEW_CPU_TRACE ARTIFACT DATA OUTPUT PREFIX LAYER POSITIONS_JSON
CPU-only diagnostic, including QSA selection; never tensor feedback or acceptance.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

gpu_path, producer, source, cpu_dir, artifact, data, output, prefix = map(Path, sys.argv[1:9])
layer, positions = int(sys.argv[9]), json.loads(sys.argv[10])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
gpu_bytes = gpu_path.read_bytes()
gpu = json.loads(gpu_bytes)
assert gpu['boundaryCaptureCompleted'] and gpu['layer'] == layer
assert gpu['normalModelQualified'] is False
assert gpu['manifestSha256'] == sha((artifact / 'manifest.json').read_bytes())
helper = Path(__file__).with_name('prepare-f32-attention-trace.py')
sys.argv = [str(helper), str(producer), str(source), str(cpu_dir), str(artifact),
            str(data), str(prefix), str(layer), json.dumps(positions)]
scope = runpy.run_path(str(helper), run_name='__main__')
environment = scope['environment']['environment']
for hook in environment['hooks']:
    hook.remove()
# The observer wrapped the eager core, without changing it. Restore the original
# before replay, so existing observation files cannot be overwritten by hooks.
environment['namespace']['eager_attention_forward'] = environment['trace_original_core']
module = environment['layer'].self_attn
cpu_bytes = (cpu_dir / 'trace.json').read_bytes()
cpu = json.loads(cpu_bytes)
assert cpu['sourceRouterInputReplayBitExact']
assert gpu['manifestSha256'] == cpu['manifestSha256']
assert gpu['inputsSha256'] == cpu['inputsSha256']


def cpu_tensor(role, stage):
    record = next(r for r in cpu['tensors'] if r['role'] == role and r['stage'] == stage)
    file = cpu_dir / record['file']
    assert sha(file.read_bytes()) == record['sha256']
    return np.load(file, allow_pickle=False).copy()


def gpu_tensor(stage):
    record = next(r for r in gpu['tensors'] if r['stage'] == stage)
    raw = Path(record['file']).read_bytes()
    assert sha(raw) == record['sha256'] and record['dtype'] == 'f32'
    value = np.frombuffer(raw, '<f4').reshape(record['shape']).copy()
    assert np.isfinite(value).all()
    return value


source_input = cpu_tensor('attention', 'projectionInput')
source_output = cpu_tensor('attn_hc', 'branchOutput')
gpu_input, gpu_output = gpu_tensor('branchInput'), gpu_tensor('branchOutput')
assert source_input.shape == gpu_input.shape == (2304, environment['cfg'].hidden_size)
assert gpu_output.shape == gpu_input.shape
selected = {}


def indexer_observer(_module, _arguments, result):
    selected['rows'] = [torch.nonzero(result[0, 0, p] == 0).flatten().tolist() for p in positions]


indexer_hook = module.indexer.register_forward_hook(indexer_observer)
try:
    with torch.no_grad():
        def replay(value):
            return module(torch.from_numpy(value).unsqueeze(0),
                          position_embeddings=environment['positional']['long'],
                          attention_mask=environment['masks']['long'], past_key_values=None)[0][0].numpy().copy()
        source_replay = replay(source_input)
        source_selections = selected['rows']
        assert np.array_equal(source_replay[positions].view('u4'), source_output.view('u4')), 'Independent source geometry replay'
        on_gpu_input = replay(gpu_input)
        gpu_input_selections = selected['rows']
        repeated = replay(gpu_input)
        assert np.array_equal(repeated.view('u4'), on_gpu_input.view('u4')), 'CPU replay must be bit-stable'
finally:
    indexer_hook.remove()


def metrics(actual, reference):
    assert actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    delta = actual.astype('f8') - reference.astype('f8')
    rms = float(np.sqrt(np.mean(reference.astype('f8') ** 2)))
    return dict(values=int(actual.size), maxAbsolute=float(np.max(np.abs(delta))),
                differenceRms=float(np.sqrt(np.mean(delta ** 2))), sourceRms=rms,
                relativeRms=float(np.sqrt(np.mean(delta ** 2)) / rms) if rms else None)


result = dict(scope='Unchanged F32 attention including QSA on the exact full GPU branch input; not model acceptance',
              normalModelQualified=False, newGpuInvocations=0, layer=layer, positions=positions,
              sourceGeometryReplayBitExact=True, sourceGeometry=[1, 2304, environment['cfg'].hidden_size],
              sourceReplayCheckedValues=int(source_output.size), sameInputReplayBitStable=True,
              gpuBoundarySha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
              producerSha256=cpu['producerSha256'], scriptSha256=sha(Path(__file__).read_bytes()),
              manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
              allRows=dict(input=metrics(gpu_input, source_input),
                           localOnSameGpuInput=metrics(gpu_output, on_gpu_input)),
              perPosition=[dict(position=p, total=metrics(gpu_output[i], source_output[j]),
                                localOnSameGpuInput=metrics(gpu_output[i], on_gpu_input[i]),
                                propagatedInput=metrics(on_gpu_input[i], source_output[j]),
                                sourceSelectionCount=len(source_selections[j]),
                                sameInputSourceSelectionCount=len(gpu_input_selections[j]),
                                addedSelections=sorted(set(gpu_input_selections[j]) - set(source_selections[j])),
                                removedSelections=sorted(set(source_selections[j]) - set(gpu_input_selections[j])))
                           for j, (p, i) in enumerate(zip(positions, positions))],
              runtimeChanged=False, acceptanceChanged=False)
assert not output.exists()
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
