"""Unchanged eager MoE on captured learned-weight GPU inputs, CPU only.

Usage: python3 -P check-moe-same-input.py <gpu.json> <run.json> <producer.py>
       <source-weights-dir> <reference-data-dir> <artifact-dir> <normal-gpu.json>
       <pinned-cpu-trace.json> <new-output-dir>
GPU routing substitution is an isolated CPU operator control, never a model
reference replacement or tensor feedback into GPU execution.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
io = runpy.run_path(str(HERE / 'verified_tensor_io.py'))
metrics = runpy.run_path(str(HERE / 'compare-upstream-stages.py'))['error_metrics']
read_tensor, load_verified_trace = io['read_tensor'], io['load_verified_trace']
gpu_file, run_file, producer, source, data, artifact, normal_file, pinned_file, out = map(Path, sys.argv[1:])
if out.exists():
    raise ValueError('Refuse to overwrite retained MoE controls')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
gpu_raw, run_raw, pinned_raw = gpu_file.read_bytes(), run_file.read_bytes(), pinned_file.read_bytes()
gpu, run, pinned = json.loads(gpu_raw), json.loads(run_raw), json.loads(pinned_raw)
assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert gpu['runSha256'] == sha(run_raw) and run['captureMoe'] is True
assert gpu['manifestSha256'] == pinned['manifestSha256'] == sha((artifact / 'manifest.json').read_bytes())
assert gpu['inputsSha256'] == pinned['inputsSha256'] == sha((data / 'learned-forward-inputs.json').read_bytes())
layer = run['stopBeforeMoeLayer']
assert layer == 1, 'This retained control covers the layer-1 prefill geometry'
actual_input = read_tensor(gpu['moeInput'], gpu_file.parent).copy()
actual_output = read_tensor(gpu['moeOutput'], gpu_file.parent)
indices = read_tensor(gpu['moeRouting']['indices'], gpu_file.parent).astype(np.int64)
weights = read_tensor(gpu['moeRouting']['weights'], gpu_file.parent).copy()
assert actual_input.shape == actual_output.shape == (2304, 2560)
assert indices.shape == weights.shape == (2304, 10)
assert (indices >= 0).all() and (indices < 512).all() and (weights >= 0).all()

# Re-hash the normal capture's actual buffers before binding observed slices.
_, observed, capture_proof = load_verified_trace(gpu_file, ('layer', 'role', 'stage'))
for stage, full in [('branchInput', actual_input), ('branchOutput', actual_output)]:
    record = observed[(layer, 'mlp_hc', stage)]
    assert record['positions'] == [2135]
    assert np.array_equal(read_tensor(record, gpu_file.parent), full[[2135]]), 'Full MoE data must match its observed HC slice'
normal, prior, normal_proof = load_verified_trace(normal_file, ('layer', 'role', 'stage'))
assert normal['manifestSha256'] == gpu['manifestSha256'] and normal['inputsSha256'] == gpu['inputsSha256']
bindings = []
for key, record in observed.items():
    if key not in prior:
        continue
    expected = prior[key]
    assert all(record.get(field) == expected.get(field) for field in ('sha256', 'shape', 'positions')), key
    bindings.append(list(key))
expected_bindings = {key for key in prior if key[0] in (0, 1) and key[1] in ('attn_hc', 'mlp_hc')}
assert {tuple(key) for key in bindings} == expected_bindings and expected_bindings, 'Every retained HC slice in both layers must match'

# Reuse the receipt-bound independent artifact reader and unchanged upstream
# classes. Stop setup before inputs/forward; no layer or reference is rewritten.
original = producer.read_bytes()
assert sha(original) == pinned['producerSha256']
text = original.decode()
anchor = 'ROOT = Path(__file__).resolve().parent'
assert text.count(anchor) == 1
text = text.replace(anchor, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
anchor = '# ---- Inputs '
assert text.count(anchor) == 1
text = text[:text.index(anchor)]
sys.argv = [str(producer), str(source), str(out), '--dtype', 'float32', '--artifact', str(artifact)]
env = {'__name__': '__main__', '__file__': str(producer)}
exec(compile(text, str(producer) + ':moe-setup-only', 'exec'), env)
assert torch.get_float32_matmul_precision() == 'highest'
module = env['namespace']['Qwen4ExpTextSparseMoeBlock'](env['cfg']).eval()
env['load'](module, f'model.language_model.layers.{layer}.mlp.')
x = torch.from_numpy(actual_input).unsqueeze(0)
routes = {}


def observe_routes(_module, _inputs, result):
    routes['indices'] = result[2].detach().numpy().copy()
    routes['weights'] = result[1].detach().numpy().copy()


handle = module.gate.register_forward_hook(observe_routes)
with torch.no_grad():
    ordinary = module(x).squeeze(0).numpy().copy()
    repeated = module(x).squeeze(0).numpy().copy()
handle.remove()
assert np.array_equal(ordinary, repeated), 'Original CPU operator must be deterministic'


def use_observed_routes(_module, _inputs, result):
    return result[0], torch.from_numpy(weights), torch.from_numpy(indices)


handle = module.gate.register_forward_hook(use_observed_routes)
try:
    with torch.no_grad():
        forced = module(x).squeeze(0).numpy().copy()
finally:
    handle.remove()
with torch.no_grad():
    flat = x.squeeze(0)
    shared = (torch.sigmoid(module.shared_expert_gate(flat)) * module.shared_expert(flat)).numpy().copy()

policy_path = producer.parents[2] / 'src/config/evidence/boundary-evidence-policy.json'
policy_bytes = policy_path.read_bytes()
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = json.loads(policy_bytes)['tolerancePolicies'][policy_id]


def compare(actual, expected):
    result = metrics(actual, expected)
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    result['worstPositionAndColumn'] = list(map(int, np.unravel_index(diff.argmax(), diff.shape)))
    result['outsideExistingPolicy'] = int(np.count_nonzero(diff > policy['absolute'] + policy['relative'] * np.abs(expected)))
    result['checkedValues'] = int(diff.size)
    result['position2135'] = metrics(actual[2135], expected[2135])
    return result


report = dict(scope='Layer-1 learned MoE same-input prefill controls; CPU-only routing substitutions, not model qualification',
              normalModelQualified=False, layer=layer, torch=torch.__version__, threads=torch.get_num_threads(),
              gpuTraceSha256=sha(gpu_raw), runSha256=sha(run_raw), pinnedCpuTraceSha256=sha(pinned_raw),
              manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
              producerSha256=sha(original), setupSha256=sha(text.encode()), scriptSha256=sha(Path(__file__).read_bytes()),
              tensorReaderSha256=sha((HERE / 'verified_tensor_io.py').read_bytes()),
              tensorVerification=dict(capture=capture_proof, normal=normal_proof), normalBindings=bindings,
              policy=dict(id=policy_id, sha256=sha(policy_bytes), **policy),
              sourceRepeatBitExact=True, sourceRouterComparedPositions=2304,
              sourceRouterExpertSetDifferences=int(np.count_nonzero(np.any(np.sort(indices, axis=1) != np.sort(routes['indices'], axis=1), axis=1))),
              sourceRouterWeightDifference=metrics(weights, routes['weights']),
              unchangedSourceOnGpuInput=compare(actual_output, ordinary),
              sourceExpertsWithGpuRoutesAndWeights=compare(actual_output, forced),
              omitSharedExpertNegativeControl=compare(forced - shared, forced),
              outputs={})
report['componentControlPassed'] = all(report[key]['outsideExistingPolicy'] == 0 for key in
    ('unchangedSourceOnGpuInput', 'sourceExpertsWithGpuRoutesAndWeights')) and report['omitSharedExpertNegativeControl']['outsideExistingPolicy'] > 0
for name, values in [('source', ordinary), ('source-gpu-routes', forced)]:
    file = out / (name + '.npy')
    np.save(file, values, allow_pickle=False)
    report['outputs'][name] = dict(file=file.name, sha256=sha(file.read_bytes()), shape=list(values.shape))
(out / 'controls.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
assert report['componentControlPassed'], 'MoE operator controls failed; retained metrics do not qualify the model'
print(json.dumps({key: report[key] for key in ('sourceRepeatBitExact', 'sourceRouterExpertSetDifferences',
    'unchangedSourceOnGpuInput', 'sourceExpertsWithGpuRoutesAndWeights', 'omitSharedExpertNegativeControl')}, indent=2))
