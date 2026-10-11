"""Check whether two retained targets admit any common 0.001 result; no activation."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)


root = Path('artifacts/whole-model-f64-complete-20261011')
proposal = read(root / 'proposal.json')
index = read(root / 'independent-output-index.json')
contracts = [read(entry['path']) for entry in proposal['contracts']]
for entry in proposal['contracts']:
    assert digest(entry['path']) == entry['sha256']
historical_input = contracts[0]['inputs']['historicalReference']
assert all(contract['inputs']['historicalReference'] == historical_input for contract in contracts)
assert digest(historical_input['path']) == historical_input['sha256']
historical = read(historical_input['path'])
tolerance = contracts[0]['tolerance']
assert tolerance == 0.001 and all(contract['tolerance'] == tolerance for contract in contracts)
rows, seen = [], set()
for shard in index['shards']:
    assert digest(shard['path']) == shard['sha256']
    report = read(shard['path'])
    assert report['toolSha256'] == proposal['toolSha256']
    assert report['contractSha256'] == proposal['contracts'][shard['index']]['sha256']
    assert [row['step'] for row in report['results']] == shard['steps']
    for row in report['results']:
        request, step = row['index'], row['step']
        assert request == shard['index'] and (request, step) not in seen
        seen.add((request, step))
        precise = np.frombuffer(base64.b64decode(row['logitsF64'], validate=True), '<f8')
        frozen = np.frombuffer(base64.b64decode(historical['expected'][request]['steps'][step]['logits'], validate=True), '<f4').astype(np.float64)
        assert precise.shape == frozen.shape and np.isfinite(precise).all() and np.isfinite(frozen).all()
        gaps = np.abs(precise - frozen)
        token = int(gaps.argmax())
        left, right = sorted([float(precise[token]), float(frozen[token])])
        rows.append({'request': request, 'step': step, 'tokenId': token,
                     'independentF64': float(precise[token]), 'historicalF32': float(frozen[token]),
                     'maximumSeparation': float(gaps[token]),
                     'incompatibleLogits': int((gaps > 2 * tolerance).sum()),
                     'intervalGap': max(0.0, right - tolerance - (left + tolerance)),
                     'bestPossibleMaximumErrorToBoth': float(gaps[token]) / 2})
assert seen == {(index, step) for index, contract in enumerate(contracts) for step in contract['steps']}
assert len(rows) == 55
result = {'scope': 'Compatibility of retained historical and independent mathematical targets; not runtime qualification or reference activation',
          'referenceActivated': False, 'toleranceUnchanged': tolerance,
          'independentIndexSha256': digest(root / 'independent-output-index.json'),
          'historicalReference': historical_input, 'checkerSha256': digest(__file__),
          'prefixes': len(rows), 'incompatiblePrefixes': sum(row['incompatibleLogits'] > 0 for row in rows),
          'incompatibleLogits': sum(row['incompatibleLogits'] for row in rows),
          'worstWitness': max(rows, key=lambda row: row['maximumSeparation']), 'rows': rows}
Path(__file__).with_name('result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ['prefixes', 'incompatiblePrefixes', 'incompatibleLogits', 'worstWitness']}))
