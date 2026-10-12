"""Export one actual decode projection; independent GGUF dequantization and F64 dot."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from gguf import dequantize, GGMLQuantizationType

capture_path, prefill_path, destination = map(Path, sys.argv[1:])
capture = json.loads(capture_path.read_bytes())
assert capture['observation']['maximumUnmaskedDifference'] == 0
spec = importlib.util.spec_from_file_location('audit', 'tools/qwen35-short-prefix-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
prefill = json.loads(prefill_path.read_bytes())
gpu = audit.read_capture(capture, prefill['M'], capture['observationTarget']['step'])
raw = base64.b64decode(prefill['weights'])
assert hashlib.sha256(raw).hexdigest() == prefill['weightsSha256']
weights = dequantize(np.frombuffer(raw, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
weights = weights.reshape(prefill['N'], prefill['K']).astype(np.float64)
input = gpu['layer.0.attn.post_input_norm'][-1:]
output = gpu['layer.0.attn.qkv_proj'][-1:]
selected = [row for row in capture['quantizedDispatch'] if row['entry'] == 'main_gemv']
assert len(selected) == 1
selected = selected[0]
assert selected['shaderSha256'] == hashlib.sha256(Path('src/gpu/kernels/fused_matmul_q4.wgsl').read_bytes()).hexdigest()
constants = selected['constants']
workgroups = [prefill['N'] // constants['COLS_PER_WG'], 1, 1]
assert selected['dispatches'][0]['dimensions'] == workgroups
encode = lambda value: base64.b64encode(value).decode()
input_bytes = input.astype('<f4').tobytes()
fixture = {**prefill, 'M': 1, 'input': encode(input_bytes),
    'inputSha256': hashlib.sha256(input_bytes).hexdigest(),
    'oracleFloat64': encode((input @ weights.T).astype('<f8').tobytes()),
    'baseline026': encode(output.astype('<f4').tobytes()),
    'captureSha256': hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    'step': capture['observationTarget']['step'],
    'execution': {'entryPoint': selected['entry'], 'constants': constants,
                  'workgroups': workgroups, 'features': ['subgroups']},
    'capturedShaderSha256': selected['shaderSha256']}
destination.write_text(json.dumps(fixture) + '\n')
