"""Compact, committed summary of a learned-forward reference run (the full JSON stays on the models drive).

Keeps per-position argmax, top-1 margins and top-5 ids, teacher-forced accuracy on the long input,
near-tie counts (router, QSA blocks, logits), decay ranges, and the SHA-256 of the full outputs.
Usage: python3 -I summarize-learned-forward-reference.py <learned-forward-reference-<dtype>.json> [<logits.npy> ...]
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
full_path = Path(sys.argv[1])
full = json.loads(full_path.read_text())
inputs = {s['id']: s['tokens'] for s in json.loads((ROOT / 'learned-forward-inputs.json').read_text())['sequences']}
dense_limit = 2051
out = {'schema': 'doppler.qwen38-learned-forward-summary/v1', 'dtype': full['dtype'], 'torch': full['torch'],
       'checkpoint': full['checkpoint'], 'upstream': full['upstream'], 'surface': full['surface'],
       'fullReference': {'file': full_path.name, 'sha256': hashlib.sha256(full_path.read_bytes()).hexdigest()},
       'logitFiles': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:]},
       'tableRowsRead': full['tableRowsRead'], 'layerSeconds': full['layerSeconds'], 'sequences': {}}
for sid, seq in full['sequences'].items():
    tokens = inputs[sid]
    nxt = range(len(tokens) - 1)
    hit = [seq['argmax'][p] == tokens[p + 1] for p in nxt]
    qsa = [v for layer in seq['qsa'].values() for v in layer.values()]
    out['sequences'][sid] = {
        'tokens': len(tokens), 'argmax': seq['argmax'], 'top5Ids': [ids[:5] for ids in seq['top20Ids']],
        'top1Margin': [round(m, 6) for m in seq['top1Margin']],
        'teacherForcedTop1': {'all': sum(hit) / max(1, len(hit)),
                              'dense': sum(hit[:dense_limit - 1]) / max(1, min(len(hit), dense_limit - 1)),
                              'sparse': (sum(hit[dense_limit:]) / len(hit[dense_limit:])) if len(hit) > dense_limit else None},
        'nearTies': {'logitTop1Below0.01': sum(1 for m in seq['top1Margin'] if m < 0.01),
                     'routerMarginBelow1e-3': sum(1 for layer in seq['router'].values() for m in layer['marginKthToNext'] if m < 1e-3),
                     'routerExactTies': sum(1 for layer in seq['router'].values() for m in layer['marginKthToNext'] if m == 0),
                     'qsaBlockMarginMin': min((v['blockMargin'] for v in qsa if v['blockMargin'] is not None), default=None)},
        'decayRange': [min(v['min'] for v in seq['decay'].values()), max(v['max'] for v in seq['decay'].values())],
        'qsaSampledQueries': {layer: sorted(map(int, v)) for layer, v in seq['qsa'].items()},
    }
target = ROOT / f"learned-forward-summary-{full['dtype']}.json"
target.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(json.dumps({k: v['teacherForcedTop1'] | v['nearTies'] for k, v in out['sequences'].items()}, indent=1), target.stat().st_size)
