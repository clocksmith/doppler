"""Synthetic controls for the independent attention diagnostic, not model acceptance."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


spec = importlib.util.spec_from_file_location(
    'deployed_reference', Path(__file__).parents[2] / 'tools/qwen35-deployed-reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

# Closed-form normalization and batch independence, including a partial lane.
for width in [1, 127, 256, 1025]:
    values = torch.stack((torch.full((width,), 2.0), torch.zeros(width)))
    actual = reference.tree_rms_normalize(values, 1e-6).numpy()
    np.testing.assert_allclose(actual[0], 2 / np.sqrt(4 + 1e-6), rtol=2e-7)
    np.testing.assert_array_equal(actual[1], 0)
try:
    reference.tree_rms_normalize(torch.tensor([[float('nan')]]), 1e-6)
    raise RuntimeError('Nonfinite normalization input was accepted')
except AssertionError:
    pass


def row(name, values):
    return {'opId': 'layer.0.attn.' + name, 'capture': {'data': values}}


model = SimpleNamespace(
    config=SimpleNamespace(num_key_value_heads=1, num_attention_heads=2),
    model=SimpleNamespace(layers=[SimpleNamespace(
        self_attn=SimpleNamespace(head_dim=2, scaling=2 ** -0.5))]))
cache = SimpleNamespace(layers=[SimpleNamespace(
    keys=torch.zeros((1, 1, 2, 2), dtype=torch.float16),
    values=torch.tensor([[[[2, 3], [4, 5]]]], dtype=torch.float16))])
capture = {'observation': {'timeline': [
    {'opId': 'embed.out'}, row('k_rope', [0, 0]), row('v_proj', [2, 3]),
    {'opId': 'embed.out'}, row('k_rope', [0, 0]), row('v_proj', [4, 5]),
    row('q_rope', [1, 0, 0, 1]), row('core_out', [3, 4, 3, 4]),
]}}
expected = reference.compare_attention_history(model, capture, cache)
control = reference.ObservedCacheControl(model, capture)
original_keys = torch.full((1, 1, 1, 2), 99.0)
original_values = torch.full((1, 1, 1, 2), 77.0)
keys, values = control.take(0, original_keys, original_values)
assert torch.equal(keys, torch.zeros_like(keys))
assert values.flatten().tolist() == [2, 3]
assert original_keys.flatten().tolist() == [99, 99]
assert original_values.flatten().tolist() == [77, 77]
try:
    control.assert_consumed()
    raise RuntimeError('Unconsumed cache history was accepted')
except AssertionError:
    pass
assert control.take(0, original_keys, original_values)[1].flatten().tolist() == [4, 5]
control.assert_consumed()
try:
    control.take(0, original_keys, original_values)
    raise RuntimeError('Exhausted cache history was accepted')
except AssertionError:
    pass
try:
    reference.ObservedCacheControl(model, {'observation': {'timeline': [row('k_rope', [0, 0])]}})
    raise RuntimeError('Missing cache operands were accepted')
except AssertionError:
    pass
assert expected[0]['maxAbsError'] == 0
assert all(item['differentStoredValues'] == 0 for item in expected[0]['independentlyEvolvedCache'])
assert expected[0]['cacheOnlyOutputMaxAbsDifference'] == 0
changed_cache = copy.deepcopy(cache)
changed_cache.layers[0].values += 2
changed = reference.compare_attention_history(model, capture, changed_cache)
assert changed[0]['maxAbsError'] == 0
assert changed[0]['cacheOnlyOutputMaxAbsDifference'] == 2

# Two F32 values straddle the midpoint between adjacent F16 values near one.
midpoint = np.float32(1 + 2 ** -11)
below = np.nextafter(midpoint, np.float32(-np.inf))
above = np.nextafter(midpoint, np.float32(np.inf))
rounding = reference.compare_storage_rounding(np.array([below]), np.array([above]))
assert rounding['differentStoredValues'] == rounding['adjacentStoredValues'] == 1
assert rounding['postStorageMaxAbsError'] == 2 ** -10
assert rounding['preStorageMaxAbsError'] == 2 ** -22
assert rounding['amplifiedValues'] == 1

binary = copy.deepcopy(capture)
for item in binary['observation']['timeline']:
    if 'capture' in item:
        values = np.asarray(item['capture'].pop('data'), dtype='<f4')
        item['capture'].update(encoding='base64-f32le',
                               dataBase64=base64.b64encode(values.tobytes()).decode())
assert reference.compare_attention_history(model, binary, cache) == expected

missing = copy.deepcopy(capture)
del missing['observation']['timeline'][1]
try:
    reference.compare_attention_history(model, missing, cache)
    raise RuntimeError('Incomplete cache history was accepted')
except AssertionError as error:
    assert 'complete cache history' in str(error)

nonfinite = copy.deepcopy(capture)
nonfinite['observation']['timeline'][-1]['capture']['data'][0] = float('nan')
try:
    reference.compare_attention_history(model, nonfinite, cache)
    raise RuntimeError('Nonfinite capture was accepted')
except AssertionError as error:
    assert 'core_out' in str(error)

# A scalar, two-token recurrence has a closed form: S1 = d,
# S2 = d * (1 + alpha), where alpha = decay * (1 - beta * k^2).
linear = SimpleNamespace(conv_dim=3, value_dim=1, key_dim=1, num_v_heads=1,
                         num_k_heads=1, head_k_dim=1, head_v_dim=1,
                         conv1d=SimpleNamespace(weight=torch.ones((3, 1, 1))),
                         norm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
                         A_log=torch.zeros(1), dt_bias=torch.zeros(1))
linear_model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=linear)]))
silu_one, silu_two = 1 / (1 + np.exp(-1)), 2 / (1 + np.exp(-2))
qk = silu_one / np.sqrt(silu_one ** 2 + 1e-6)
d = 0.5 * qk * silu_two
alpha = 0.5 * (1 - 0.5 * qk ** 2)
linear_capture = {'observation': {'timeline': []}}
for state in [d, d * (1 + alpha)]:
    output = state * qk
    output = output / np.sqrt(output ** 2 + 1e-6) * silu_one
    linear_capture['observation']['timeline'].extend([
        row('qkv_proj', [1, 1, 2]), row('linear_z_proj', [1]),
        row('linear_a_proj', [0]), row('linear_b_proj', [0]),
        row('linear_core_out', [output]),
    ])
assert reference.compare_linear_history(linear_model, linear_capture)['maxAbsError'] < 1e-7
second_layer = copy.deepcopy(linear)
second_layer.norm.weight *= 2
linear_model.model.layers.append(SimpleNamespace(linear_attn=second_layer))
multi_capture = copy.deepcopy(linear_capture)
for item in linear_capture['observation']['timeline']:
    item = copy.deepcopy(item)
    item['opId'] = item['opId'].replace('layer.0.', 'layer.1.')
    if item['opId'].endswith('linear_core_out'):
        item['capture']['data'] = [2 * value for value in item['capture']['data']]
    multi_capture['observation']['timeline'].append(item)
multi = reference.compare_linear_histories(linear_model, multi_capture)
assert [item['boundary'] for item in multi] == ['layer.0.attn.linear_core_out', 'layer.1.attn.linear_core_out']
assert all(item['maxAbsError'] < 2e-7 for item in multi)
assert reference.compare_linear_histories(linear_model, {'observation': {'timeline': [
    {'opId': 'layer.0.attn.linear_core_out', 'capture': None},
]}}) == []
del multi_capture['observation']['timeline'][10]
try:
    reference.compare_linear_histories(linear_model, multi_capture)
    raise RuntimeError('Incomplete later-layer recurrence history was accepted')
except AssertionError as error:
    assert 'incomplete linear history' in str(error)
del linear_capture['observation']['timeline'][0]
try:
    reference.compare_linear_history(linear_model, linear_capture)
    raise RuntimeError('Incomplete recurrence history was accepted')
except AssertionError as error:
    assert 'incomplete linear history' in str(error)

physical = {'observation': {'target': {'step': 1}, 'maximumUnmaskedDifference': 0,
                           'prefixDifferences': [{'maximumUnmaskedDifference': 0}] * 2,
                           'timeline': [], 'physicalCache': []}}
history = []
for step, values in enumerate(([[1.0004, -2.0008], [3.0, 4.0]], [[5.0, 6.0001]])):
    history.extend(values)
    timeline = physical['observation']['timeline']
    timeline.append({'opId': 'embed.out'})
    for name in ['k_rope', 'v_proj']:
        timeline.append({'opId': 'layer.3.attn.' + name, 'capture': {'data': np.array(values).reshape(-1).tolist()}})
    encoded = base64.b64encode(np.array(history, dtype='<f2').tobytes()).decode()
    physical['observation']['physicalCache'].append({
        'step': step, 'layer': 3, 'keysId': 1, 'valuesId': 2, 'seqLen': len(history),
        'shape': [len(history), 1, 2], 'dtype': 'f16', 'layout': 'contiguous',
        'byteOffset': 0, 'byteLength': len(history) * 4,
        'reads': [{'seqLen': len(history), 'keysId': 1, 'valuesId': 2}],
        'values': {name: {'encoding': 'base64-f16le', 'dataBase64': encoded} for name in ['k', 'v']}})
assert reference.compare_physical_cache(physical, [3], 1, 2)['passed']
changed = copy.deepcopy(physical)
row = changed['observation']['physicalCache'][1]
bits = np.frombuffer(base64.b64decode(row['values']['k']['dataBase64']), dtype='<u2').copy()
bits[0] += 1
row['values']['k']['dataBase64'] = base64.b64encode(bits.tobytes()).decode()
assert not reference.compare_physical_cache(changed, [3], 1, 2)['passed']
for fault in ['missing', 'offset', 'read', 'instrumentation']:
    broken = copy.deepcopy(physical)
    if fault == 'missing': broken['observation']['physicalCache'].pop()
    elif fault == 'offset': broken['observation']['physicalCache'][1]['byteOffset'] = 4
    elif fault == 'read': broken['observation']['physicalCache'][1]['reads'][0]['keysId'] = 99
    else: broken['observation']['prefixDifferences'][0]['maximumUnmaskedDifference'] = 0.01
    try:
        reference.compare_physical_cache(broken, [3], 1, 2)
        raise RuntimeError('Invalid physical capture was accepted: ' + fault)
    except AssertionError:
        pass

print(json.dumps({'constantAttentionMean': 'pass', 'binaryCaptureEquivalence': 'pass',
                  'missingHistoryRejected': 'pass', 'nonfiniteCaptureRejected': 'pass',
                  'scalarRecurrenceClosedForm': 'pass', 'missingRecurrenceHistoryRejected': 'pass',
                  'distinctLayerWeights': 'pass', 'missingLaterLayerHistoryRejected': 'pass',
                  'uncapturedOperatorsNotCompared': 'pass',
                  'cacheOperandIsolation': 'pass', 'halfRoundingAmplification': 'pass',
                  'normalizationClosedForm': 'pass', 'nonfiniteNormalizationRejected': 'pass',
                  'cacheInterventionIsolation': 'pass', 'incompleteInterventionRejected': 'pass',
                  'physicalCacheExactRoundTrip': 'pass', 'physicalCacheBitMismatchDetected': 'pass',
                  'physicalCacheMissingRejected': 'pass', 'physicalCachePositionRejected': 'pass',
                  'physicalCacheReadIdentityRejected': 'pass', 'physicalCacheInstrumentationRejected': 'pass'}, indent=2))
