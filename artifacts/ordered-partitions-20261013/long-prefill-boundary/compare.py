"""Offline observation of the unchanged independent evaluator; no runtime fallback."""
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
output = Path(sys.argv[2])
evaluator = root / 'qwen35-f64-reference.py'
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == '90376ddec6e6bf997425ee0e1ecf1d98533434ebbe3f934ff00a878dd49e8bc7'
spec = importlib.util.spec_from_file_location('frozen_reference', evaluator)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
proposal = json.loads((root / 'proposal.json').read_text())
def bound(name):
    row = proposal['files'][name]
    data = Path(row['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256'], name
    return json.loads(data)

manifest, config, pieces = bound('manifest'), bound('sourceConfig'), bound('pieceIndex')
tokens, observation = bound('tokens'), bound('observation')
weights, shards = reference.load_weights(Path(proposal['modelRoot']), manifest, pieces)
boundaries = {}
def progress(row):
    # Read the evaluator's existing layer-completion callback frame without
    # modifying its function, operands, arithmetic, or acceptance reference.
    frame = sys._getframe(1)
    assert frame.f_code is reference.evaluate.__code__
    if row['layer'] in (7, 15):
        boundaries[row['layer']] = frame.f_locals['x'].copy()
    print(json.dumps(row), flush=True)

logits, caches = reference.evaluate(tokens, weights, config['text_config'],
                                  np.array([len(tokens) - 1]), progress)
# Observed values only enter comparison after independent evaluation.
comparisons = []
for layer, independent in boundaries.items():
    observed = observation['boundaries'][str(layer)]
    values = np.frombuffer(base64.b64decode(observed['data']), dtype='<f4').astype(np.float64)
    actual = values.reshape(len(tokens), manifest['architecture']['hiddenSize'], 2).sum(axis=-1)
    difference = np.abs(actual - independent)
    comparisons.append({'layer': layer, 'maxAbsoluteError': float(difference.max()),
                        'maxIndex': list(map(int, np.unravel_index(np.argmax(difference), difference.shape))),
                        'rowMaxAbsoluteError': difference.max(axis=1).tolist(),
                        'independentSha256': hashlib.sha256(independent.astype('<f8').tobytes()).hexdigest()})
    np.save(output.parent / ('independent-layer-' + str(layer) + '.npy'), independent)
frozen = np.frombuffer(base64.b64decode(observation['referenceLogits']), dtype='<f8')
actual = np.frombuffer(base64.b64decode(observation['actualLogits']), dtype='<f4')
result = {'schema': 'doppler.long-prefill-boundary/v1', 'scope': proposal['scope'],
          'evaluatorSha256': hashlib.sha256(evaluator.read_bytes()).hexdigest(),
          'proposalSha256': hashlib.sha256((root / 'proposal.json').read_bytes()).hexdigest(),
          'numpyVersion': np.__version__, 'shards': shards, 'boundaries': comparisons,
          'independentGreedyToken': int(np.argmax(logits[0])),
          'frozenReferenceGreedyToken': int(np.argmax(frozen)),
          'independentVersusFrozenMaxAbsoluteError': float(np.max(np.abs(logits[0] - frozen))),
          'observedVersusIndependentMaxAbsoluteError': float(np.max(np.abs(logits[0] - actual)))}
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ['independentGreedyToken', 'frozenReferenceGreedyToken',
      'independentVersusFrozenMaxAbsoluteError', 'observedVersusIndependentMaxAbsoluteError']}), flush=True)
