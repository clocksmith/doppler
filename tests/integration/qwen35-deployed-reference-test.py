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

print(json.dumps({'constantAttentionMean': 'pass', 'binaryCaptureEquivalence': 'pass',
                  'missingHistoryRejected': 'pass', 'nonfiniteCaptureRejected': 'pass'}, indent=2))
