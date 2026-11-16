"""Installed public cache encodings; legacy absence is an explicit compatibility policy."""
import copy
import json
from pathlib import Path

POLICY=json.loads(Path(__file__).with_name('weight-encodings.json').read_text())


def weight_encoding(parameters):
    if not isinstance(parameters,dict):raise ValueError('Installed weight encoding contract required')
    name=parameters['weightEncoding'] if 'weightEncoding' in parameters else POLICY['legacyEncoding']
    if not isinstance(name,str) or name not in POLICY['encodings']:raise ValueError('Unsupported installed public coefficient encoding')
    value=copy.deepcopy(POLICY['encodings'][name])
    if 'quantizationBits' in parameters and parameters['quantizationBits']!=value['quantizationBits']:raise ValueError('Public coefficient precision differs from the admitted encoding')
    return dict(value,id=name)


def encoding_for_bits(bits):
    entries=[(name,value) for name,value in POLICY['encodings'].items() if type(bits) is int and value['quantizationBits']==bits]
    if len(entries)!=1:raise ValueError('Unsupported installed quantization precision')
    return weight_encoding(dict(weightEncoding=entries[0][0],quantizationBits=bits))
