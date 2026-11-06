"""CPU-only observation of a declared prefix of unchanged upstream decoder layers.

Instrument the retained learned-reference producer; do not consume GPU values or
change model math. Save complete router inputs/logits/choices and layer outputs
so a numerical discrepancy at an unsampled position cannot be hidden.
Usage: python3 -P prepare-router-prefix.py <producer.py> <source-dir> <out-dir> <artifact-dir> <reference-data-dir> <layers>
"""
import hashlib
import json
import sys
from pathlib import Path

producer, source, out, artifact, data = map(Path, sys.argv[1:6])
layers = int(sys.argv[6])
assert len(sys.argv) == 7 and 0 < layers <= 48
original = producer.read_bytes()
text = original.decode()
assert 'ARTIFACT_MANIFEST_BYTES =' in text and 'if KV_F16:' in text, 'matched artifact/F16-KV producer required'
root = 'ROOT = Path(__file__).resolve().parent'
assert text.count(root) == 1
text = text.replace(root, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
loop = 'for layer_idx in range(cfg.num_hidden_layers):'
assert text.count(loop) == 1
text = text.replace(loop, f'for layer_idx in range({layers}):')
anchor = "    rec['minMargin'] = float(min(rec['marginKthToNext']))"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '''
    np.savez(OUT / f"router-{current['layer']}.npz",
             input=(inputs[0].detach().float().reshape(-1, cfg.hidden_size).numpy()
                    if current['layer'] < 8 else np.empty((0, cfg.hidden_size), dtype=np.float32)),
             logits=logits.detach().numpy(),
             experts=top.indices[:, :cfg.num_experts_per_tok].numpy())''')
anchor = "            states[s['id']] = out"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '''
            if layer_idx < 8:
                np.save(OUT / f"layer-{layer_idx}.npy", out.detach().float().numpy(), allow_pickle=False)''')
cut = "    mixer = namespace['Qwen4ExpTextGatedResidual']"
assert text.count(cut) == 1
text = text[:text.index(cut)]
sys.argv = [str(producer), str(source), str(out), '--dtype', 'float32',
            '--artifact', str(artifact), '--sequences', 'long', '--kv-dtype', 'f16']
environment = {'__name__': '__main__', '__file__': str(producer)}
exec(compile(text, str(producer) + ':prefix-observation', 'exec'), environment)
sha = lambda data: hashlib.sha256(data).hexdigest()
receipt = {'schema': 'doppler.flash-next-router-prefix-reference/v1',
           'scope': 'CPU upstream declared layer prefix; added observations only, no GPU inputs or replacement math',
           'producerSha256': sha(original), 'instrumentedProducerSha256': sha(text.encode()),
           'instrumentSha256': sha(Path(__file__).read_bytes()),
           'manifestSha256': sha(environment['ARTIFACT_MANIFEST_BYTES']),
           'inputsSha256': sha(environment['INPUTS_PATH'].read_bytes()),
           'torch': environment['torch'].__version__, 'surface': 'CPU',
           'layers': layers, 'sourceRevision': environment['binding']['revision'],
           'files': {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.suffix in ('.npz', '.npy')}}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
