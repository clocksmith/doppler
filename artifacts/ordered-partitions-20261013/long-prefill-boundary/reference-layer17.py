"""Offline layer-17 attribution from independently evaluated layer-16 state."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
evaluator = root / 'qwen35-f64-reference.py'
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == '90376ddec6e6bf997425ee0e1ecf1d98533434ebbe3f934ff00a878dd49e8bc7'
spec = importlib.util.spec_from_file_location('reference', evaluator)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
proposal = json.loads((root / 'proposal.json').read_text())
model = Path('/var/tmp/doppler-cold-compile-model-20261013')
manifest_bytes = (model / 'manifest.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == proposal['files']['manifest']['sha256']
m = json.loads(manifest_bytes)
source = Path('artifacts/source-model-reference-20261010/source-config.json').read_bytes()
assert hashlib.sha256(source).hexdigest() == proposal['files']['sourceConfig']['sha256']
c = json.loads(source)['text_config']
assert c['layer_types'][17] == 'linear_attention'
shard_hashes = {row['path']: row['sha256'] for row in json.loads((root / 'result.json').read_text())['shards']}
shards, weights = {}, {}
for name, entry in m['tensors'].items():
    if not name.startswith('model.language_model.layers.17.'):
        continue
    spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
    for span in spans:
        index = span['shardIndex']
        if index not in shards:
            filename = m['shards'][index]['filename']
            data = (model / filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == shard_hashes[filename]
            shards[index] = data
    data = b''.join(shards[s['shardIndex']][s['offset']:s['offset'] + s['size']] for s in spans)
    assert len(data) == entry['size']
    if entry['dtype'] == 'Q4_K_M':
        assert entry['layout'] == 'row'
        array = r.dequantize(np.frombuffer(data, dtype=np.uint8).copy(), r.GGMLQuantizationType.Q4_K)
    else:
        assert entry['dtype'] in ('F16', 'F32')
        array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4')
    weights[name.removeprefix('model.language_model.layers.17.')] = array.reshape(entry['shape']).astype(np.float64)
archive = (root / 'independent-layers.npz').read_bytes()
assert hashlib.sha256(archive).hexdigest() == json.loads((root / 'independent-layers.json').read_text())['sha256']
independent = np.load(root / 'independent-layers.npz')
x = independent['layer_16'].copy()
norm = r.rms_norm(x, weights['input_layernorm.weight'], c['rms_norm_eps'])
w = lambda key: weights['linear_attn.' + key]
core = r.delta_history(norm @ w('in_proj_qkv.weight').T, norm @ w('in_proj_z.weight').T,
                      norm @ w('in_proj_a.weight').T, norm @ w('in_proj_b.weight').T,
                      w('conv1d.weight').squeeze(1), w('A_log'), w('dt_bias'), w('norm.weight'), c)
update = core @ w('out_proj.weight').T
post = x + update
ffn_norm = r.rms_norm(post, weights['post_attention_layernorm.weight'], c['rms_norm_eps'])
gate = ffn_norm @ weights['mlp.gate_proj.weight'].T
up = ffn_norm @ weights['mlp.up_proj.weight'].T
ffn = (gate * r.sigmoid(gate) * up) @ weights['mlp.down_proj.weight'].T
out = post + ffn
difference = float(np.max(np.abs(out - independent['layer_17'])))
print(json.dumps({'localVsRetainedMaxAbsoluteError': difference}), flush=True)
assert difference < 2e-11
np.savez_compressed(root / 'reference-layer17.npz', **{'attn.post_input_norm': norm,
 'attn.linear_core_out': core, 'attn.out': update, 'attn.post_attn': post,
 'ffn.normed': ffn_norm, 'ffn.out': ffn, 'layer.out': out})
result = {'scope': 'Offline attribution only; independent layer-16 input, original evaluator equations and weights',
 'sha256': hashlib.sha256((root / 'reference-layer17.npz').read_bytes()).hexdigest(),
 'localVsRetainedMaxAbsoluteError': difference, 'localNumpyVersion': np.__version__, 'operatorDiagnosticTolerance': 2e-11, 'shards': [m['shards'][i]['filename'] for i in shards],
 'referenceInputSha256': hashlib.sha256(archive).hexdigest()}
(root / 'reference-layer17.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
