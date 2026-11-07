"""Observe unchanged F32 upstream routing and bounded periodic checkpoints.

Usage: python3 prepare-router-prefix.py PRODUCER SOURCE OUT ARTIFACT DATA INPUTS REFERENCE
CPU-only observation. Never imports or substitutes GPU values.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


producer, source, output, artifact, data, inputs, reference = map(Path, sys.argv[1:])
require(not output.exists(), 'Retain prior attempts; choose a fresh output directory')
require(shutil.disk_usage(output.parent).free > 12 * 1024**3, 'Need the 4 GiB trace allowance plus 8 GiB headroom')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
original = producer.read_bytes()
text = original.decode()


def replace_once(anchor, replacement):
    global text
    require(text.count(anchor) == 1, f'Ambiguous producer boundary: {anchor}')
    text = text.replace(anchor, replacement)


replace_once('ROOT = Path(__file__).resolve().parent', f'ROOT = Path({str(data.resolve())!r})')
anchor = "    rec['minMargin'] = float(min(rec['marginKthToNext']))"
replace_once(anchor, anchor+'''
    np.savez(OUT / f"router-{current['layer']}.npz",
             input=inputs[0].detach().float().reshape(-1, cfg.hidden_size).numpy(),
             logits=logits.detach().numpy(), experts=output[2].detach().numpy(),
             scores=output[1].detach().float().numpy())''')
anchor = "            states[s['id']] = out"
replace_once(anchor, anchor+'''
            if layer_idx == 2 or layer_idx % 4 == 3:
                np.save(OUT / f"layer-{layer_idx}.npy", out.detach().float().numpy(), allow_pickle=False)''')
cut = "    mixer = namespace['Qwen4ExpTextGatedResidual']"
require(text.count(cut) == 1, 'Expected one root mixer boundary')
text = text[:text.index(cut)]
sys.argv = [str(producer), str(source), str(output), '--dtype', 'float32', '--artifact', str(artifact),
            '--inputs', str(inputs), '--sequences', 'long']
environment = {'__name__': '__main__', '__file__': str(producer)}
exec(compile(text, str(producer)+':f32-routing-observation', 'exec'), environment)
import numpy as np
baseline_bytes = reference.read_bytes()
baseline = json.loads(baseline_bytes)
require(baseline['producerSha256'] == sha(original), 'Different baseline producer')
require(baseline['inputsSha256'] == sha(inputs.read_bytes()), 'Different reference input')
require(baseline['artifactManifestSha256'] == sha((artifact/'manifest.json').read_bytes()), 'Different artifact')
records = environment['records']['long']
expected = baseline['sequences']['long']
for layer in range(48):
    actual = np.asarray(records['layers'][str(layer)]['samples'], dtype='<f4')
    target = np.asarray(expected['layers'][str(layer)]['samples'], dtype='<f4')
    require(np.array_equal(actual.view('u4'), target.view('u4')), f'Observed CPU trace diverged from reference at {layer}')
    require(records['router'][str(layer)] == expected['router'][str(layer)], f'Router observation differs at {layer}')
receipt = dict(scope='Unchanged upstream F32 routing observation; not model acceptance', completed=True,
               kvStorageDtype='f32', producerSha256=sha(original), instrumentedProducerSha256=sha(text.encode()),
               instrumentationSha256=sha(Path(__file__).read_bytes()), referenceSha256=sha(baseline_bytes),
               manifestSha256=sha((artifact/'manifest.json').read_bytes()), inputsSha256=sha(inputs.read_bytes()),
               samplesAndRouterMetadataMatchReference=True,
               files={p.name:sha(p.read_bytes()) for p in output.iterdir() if p.suffix in ['.npy','.npz']})
(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(completed=True, files=len(receipt['files']))))
