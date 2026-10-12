"""Freeze actual normalization operands and independent equations for replay."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

capture_path, destination = map(Path, sys.argv[1:3])
qk_layer = int(sys.argv[3]) if len(sys.argv) > 3 else None
capture = json.loads(capture_path.read_bytes())
assert capture['observation']['maximumUnmaskedDifference'] == 0
spec = importlib.util.spec_from_file_location('audit', 'tools/qwen35-short-prefix-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
gpu = audit.read_capture(capture, 184, capture["observationTarget"]["step"] if qk_layer is not None else 9)
root = Path('models/local/qwen-3-5-0-8b-q4k-ehaf16')
manifest = json.loads((root / 'manifest.json').read_bytes())
if qk_layer is not None:
    entry = manifest['tensors'][f'model.language_model.layers.{qk_layer}.self_attn.k_norm.weight']
    assert entry['dtype'] == 'F16'
    raw = (root / manifest['shards'][entry['shard']]['filename']).read_bytes()[entry['offset']:entry['offset'] + entry['size']]
    width = manifest['architecture']['headDim']
    values = gpu[f'layer.{qk_layer}.attn.k_proj'].reshape(-1).tolist()
    fixture = {'headDim': width, 'q': values, 'k': values,
        'qWeight': base64.b64encode(raw).decode(), 'kWeight': base64.b64encode(raw).decode(),
        'epsilon': manifest['inference']['normalization']['rmsNormEps'],
        'constants': {'RMS_NORM_OFFSET': True, 'Q_WEIGHT_IS_F16': True, 'K_WEIGHT_IS_F16': True},
        'captureSha256': hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        'scope': 'Captured K operands duplicated into independent Q/K rows for normalization/store control',
        'operation': f'layer.{qk_layer}.attn.k_norm'}
    destination.write_text(json.dumps(fixture) + '\n')
    sys.exit(0)
entry = manifest['tensors']['model.language_model.layers.0.input_layernorm.weight']
assert entry['dtype'] == 'F16'
raw = (root / manifest['shards'][entry['shard']]['filename']).read_bytes()[entry['offset']:entry['offset'] + entry['size']]
x = gpu['embed.out']
weight = np.frombuffer(raw, dtype='<f2').astype(np.float64)
policy = manifest['inference']['normalization']
assert policy['rmsNormWeightOffset'] is True
expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + policy['rmsNormEps']) * (1 + weight)
selected = [row for row in capture['normalizationDispatch'] if row['constants']['PRE_RESIDUAL'] == 0]
assert len(selected) == 1
assert selected[0]['shaderSha256'] == hashlib.sha256(Path('src/gpu/kernels/rmsnorm.wgsl').read_bytes()).hexdigest()
encode = lambda x: base64.b64encode(x).decode()
fixture = {'input': encode(x.astype('<f4').tobytes()), 'weight': encode(raw),
           'expected': encode(expected.astype('<f8').tobytes()),
           'observed': encode(gpu['layer.0.attn.post_input_norm'].astype('<f4').tobytes()),
           'rows': len(x), 'width': x.shape[1], 'epsilon': policy['rmsNormEps'],
           'pipeline': selected[0], 'captureSha256': hashlib.sha256(capture_path.read_bytes()).hexdigest()}
destination.write_text(json.dumps(fixture) + '\n')
