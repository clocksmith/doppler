"""Create bounded synthetic initializer readback fixtures, never model weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto as T
from binding import digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
policy_path = Path(__file__).with_name('readback-micro.json')
policy = json.loads(policy_path.read_text())
args.out.mkdir(parents=True, exist_ok=False)
cases = []
for size in policy['fixtureBytes']:
    if size > policy['maxAllocationBytes'] or size % (policy['rowWidth'] * 4):
        raise ValueError('synthetic fixture exceeds explicit allocation or row geometry')
    count = size // 4
    values = ((np.arange(count, dtype=np.uint32) % policy['patternPeriod']).astype(np.float32) /
              np.float32(policy['patternPeriod']) - np.float32(0.5)).reshape(-1, policy['rowWidth'])
    graph = helper.make_graph([helper.make_node('Gather', ['fixture', 'ids'], ['selected'], axis=0),
        helper.make_node('Identity', ['fixture'], ['observed'])], 'synthetic_initializer_readback',
        [helper.make_tensor_value_info('ids', T.INT64, [1])],
        [helper.make_tensor_value_info('selected', T.FLOAT, [1, policy['rowWidth']]),
         helper.make_tensor_value_info('observed', T.FLOAT, list(values.shape))],
        [numpy_helper.from_array(values, 'fixture')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 23)], ir_version=10)
    filename = 'synthetic-' + str(size) + '.onnx'
    onnx.checker.check_model(model)
    onnx.save(model, args.out / filename)
    cases.append({'file': filename, 'bytes': size, 'shape': list(values.shape),
        'graphSha256': digest(args.out / filename), 'expectedSha256': hashlib.sha256(values.tobytes()).hexdigest()})
(args.out / 'fixtures.json').write_text(json.dumps({'policy': policy, 'policySha256': digest(policy_path),
    'sourceSha256': digest(Path(__file__)), 'cases': cases}, indent=2) + '\n')
