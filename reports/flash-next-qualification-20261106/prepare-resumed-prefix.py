"""Observe an unchanged upstream CPU suffix from a receipt-bound CPU checkpoint.

Usage: python3 -P prepare-resumed-prefix.py <producer> <source> <out> <artifact>
       <reference-data> <prefix-dir> <first-layer> <exclusive-last-layer> [<qsa-position>]
"""
import hashlib
import json
import sys
from pathlib import Path

assert len(sys.argv) in (9, 10)
producer, source, output, artifact, data, prefix = map(Path, sys.argv[1:7])
first, last = map(int, sys.argv[7:9])
qsa_position = int(sys.argv[9]) if len(sys.argv) == 10 else 2129
assert 0 <= qsa_position < 2304
assert 0 < first < last <= 48 and not output.exists()
sha = lambda b: hashlib.sha256(b).hexdigest()
parent_bytes = (prefix / 'receipt.json').read_bytes()
parent = json.loads(parent_bytes)
original = producer.read_bytes()
assert sha(original) == parent['producerSha256']
checkpoint = prefix / f'layer-{first - 1}.npy'
assert sha(checkpoint.read_bytes()) == parent['files'][checkpoint.name]
assert sha((artifact / 'manifest.json').read_bytes()) == parent['manifestSha256']
text = original.decode()
anchor = 'ROOT = Path(__file__).resolve().parent'
assert text.count(anchor) == 1
text = text.replace(anchor, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
anchor = 'with torch.no_grad():\n    for layer_idx in range(cfg.num_hidden_layers):'
assert text.count(anchor) == 1
text = text.replace(anchor, "assert len(sequences) == 1 and sequences[0]['id'] == 'long'\n"
                    + "states['long'] = torch.from_numpy(np.load(" + repr(str(checkpoint.resolve()))
                    + ", allow_pickle=False)).to(DTYPE)\nwith torch.no_grad():\n"
                    + f'    for layer_idx in range({first}, {last}):')
anchor = "    rec['minMargin'] = float(min(rec['marginKthToNext']))"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '''
    np.savez(OUT / f"router-{current['layer']}.npz",
             input=inputs[0].detach().float().reshape(-1, cfg.hidden_size).numpy(),
             logits=logits.detach().numpy(), experts=output[2].detach().numpy(),
             scores=output[1].detach().float().numpy())''')
anchor = 'sparse = [q for q in sample_positions(T) if q + 1 > module.token_budget + module.compress_ratio - 1]'
assert text.count(anchor) == 1
text = text.replace(anchor, f'sparse = [q for q in sorted(set(sample_positions(T) + [{qsa_position}])) if q < T and q + 1 > module.token_budget + module.compress_ratio - 1]')
anchor = "                           'completeBlocks': blocks}"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '''
        if query == TRACE_QSA_POSITION:
            np.savez(OUT / f"qsa-{current['layer']}-{query}.npz",
                     query=q[0, query].detach().float().numpy(),
                     rawKeys=raw[0].detach().float().numpy(), pooledKeys=keys.detach().float().numpy(),
                     scores=scores.detach().float().numpy(), selected=np.asarray(selected, dtype=np.int32))''')
anchor = "            states[s['id']] = out"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '''
            np.save(OUT / f"layer-{layer_idx}.npy", out.detach().float().numpy(), allow_pickle=False)''')
cut = "    mixer = namespace['Qwen4ExpTextGatedResidual']"
assert text.count(cut) == 1
text = text[:text.index(cut)]
sys.argv = [str(producer), str(source), str(output), '--dtype', 'float32',
            '--artifact', str(artifact), '--sequences', 'long', '--kv-dtype', 'f16']
environment = {'__name__': '__main__', '__file__': str(producer), 'TRACE_QSA_POSITION': qsa_position}
exec(compile(text, str(producer) + ':resumed-prefix-observation', 'exec'), environment)
assert sha(environment['INPUTS_PATH'].read_bytes()) == parent['inputsSha256']
receipt = dict(schema='doppler.flash-next-resumed-cpu-prefix/v1',
               scope='Unchanged upstream CPU suffix from a frozen CPU checkpoint, with observations only; no GPU values or model qualification',
               producerSha256=sha(original), instrumentedProducerSha256=sha(text.encode()),
               instrumentSha256=sha(Path(__file__).read_bytes()),
               manifestSha256=sha(environment['ARTIFACT_MANIFEST_BYTES']), inputsSha256=parent['inputsSha256'],
               parentReceiptSha256=sha(parent_bytes), checkpointSha256=parent['files'][checkpoint.name],
               firstLayer=first, exclusiveLastLayer=last, qsaPosition=qsa_position, torch=environment['torch'].__version__,
               files={p.name: sha(p.read_bytes()) for p in output.iterdir() if p.suffix in ('.npz', '.npy')},
               records=environment['records'])
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(firstLayer=first, exclusiveLastLayer=last, completed=True)))
