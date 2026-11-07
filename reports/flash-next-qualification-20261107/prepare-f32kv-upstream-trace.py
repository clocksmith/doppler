"""Reuse retained observation hooks on a bounded native-F32 CPU suffix.

Arguments match prepare-layer0-upstream-trace.py: PRODUCER SOURCE OUT ARTIFACT
DATA PREFIX TARGET_LAYER POSITIONS_JSON. The prior F16 instrument is unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


sha = lambda raw: hashlib.sha256(raw).hexdigest()
producer, source, output, artifact, data, prefix = map(Path, sys.argv[1:7])
target = int(sys.argv[7])
positions = json.loads(sys.argv[8])
binding_bytes = (prefix/'receipt.json').read_bytes()
binding = json.loads(binding_bytes)
require(binding['completed'] and binding['kvStorageDtype'] == 'f32', 'Expected a completed independent F32 prefix')
available = [int(Path(name).stem.split('-')[1]) for name in binding['files'] if name.startswith('layer-') and name.endswith('.npy')]
previous = max(i for i in available if i < target)
require(target-previous <= 4, 'This control admits at most four resumed layers')
base = Path(__file__).resolve().parents[1]/'flash-next-qualification-20261106/prepare-layer0-upstream-trace.py'
original = base.read_bytes()
text = original.decode()


def replace_once(anchor, replacement):
    global text
    require(text.count(anchor) == 1, f'Ambiguous observation boundary: {anchor}')
    text = text.replace(anchor, replacement)


replace_once("checkpoint = prefix / f'layer-{target_layer - 1}.npy'", f"checkpoint = prefix / 'layer-{previous}.npy'")
replace_once("f'for layer_idx in range({target_layer}, {target_layer + 1}):'", f"f'for layer_idx in range({previous+1}, {{target_layer + 1}}):'")
replace_once("if role == 'mlp_hc': raise TRACE_STOP", "if role == 'mlp_hc' and layer_idx == TRACE_TARGET: raise TRACE_STOP")
replace_once("'TRACE_POSITIONS': positions}", "'TRACE_POSITIONS': positions, 'TRACE_TARGET': target_layer}")
replace_once("'--sequences', 'long', '--kv-dtype', 'f16'", "'--sequences', 'long'")
# This instrument's full-attention branch is F16-specific. Refuse to silently
# claim coverage of it: the current first differing layer is linear layer 25.
replace_once("assert KV_F16", "raise ValueError('This bounded F32 observation currently admits linear layers only')")
environment = {'__name__':'__main__', '__file__':str(Path(__file__).resolve())}
exec(compile(text, str(base)+':native-f32-bounded-resume', 'exec'), environment)
trace_path = output/'trace.json'
trace = json.loads(trace_path.read_bytes())
require(trace['diagnosticCompleted'], 'Incomplete source trace')
row = next(r for r in trace['tensors'] if r['role'] == 'mlp_hc' and r['stage'] == 'branchInput')
require(row['positions'] == positions, 'Different observed positions')
raw = (output/row['file']).read_bytes()
require(sha(raw) == row['sha256'], 'Trace hash mismatch')
filename = f'router-{target}.npz'
require(sha((prefix/filename).read_bytes()) == binding['files'][filename], 'Reference router hash mismatch')
expected = np.load(prefix/filename, allow_pickle=False)['input'][positions]
actual = np.load(output/row['file'], allow_pickle=False)
require(np.array_equal(actual.view('u4'), expected.view('u4')), 'CPU resume must reproduce the independent router input bit-exactly')
trace.update(kvStorageDtype='f32', firstLayer=previous+1, sourceRouterInputReplayBitExact=True,
             baseInstrumentSha256=sha(original), adaptedInstrumentSha256=sha(text.encode()))
trace_path.write_text(json.dumps(trace,indent=2)+'\n')
print(json.dumps(dict(diagnosticCompleted=True, firstLayer=previous+1, targetLayer=target, sourceRouterInputReplayBitExact=True)))
