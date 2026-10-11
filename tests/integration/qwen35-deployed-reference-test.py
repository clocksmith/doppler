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

print(json.dumps({'constantAttentionMean': 'pass', 'binaryCaptureEquivalence': 'pass',
                  'missingHistoryRejected': 'pass', 'nonfiniteCaptureRejected': 'pass',
                  'scalarRecurrenceClosedForm': 'pass', 'missingRecurrenceHistoryRejected': 'pass',
                  'distinctLayerWeights': 'pass', 'missingLaterLayerHistoryRejected': 'pass',
                  'uncapturedOperatorsNotCompared': 'pass',
                  'cacheOperandIsolation': 'pass', 'halfRoundingAmplification': 'pass'}, indent=2))
