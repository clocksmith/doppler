"""Observation side-effect and binding controls; synthetic evidence, not model qualification."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from lib.source_logit_observation import SourceLogitObserver

contract = {'schema': 'doppler.source-logit-observation/v1', 'suiteSha256': 'suite', 'vocabSize': 13,
            'manifestSha256': 'manifest', 'positions': [{'caseId': 'fixture', 'seed': 23, 'steps': [0, 1, 7]}]}
torch.manual_seed(99)
model = torch.nn.Linear(4, 13)
weights = copy.deepcopy(model.state_dict())


def sample(observer=None):
    random = torch.Generator().manual_seed(23)
    context, all_logits, selected = [3, 2], [], []
    for step in range(3):
        with torch.inference_mode():
            tensor = model(torch.tensor([1., 2., float(context[-1]), 4.]))
        logits = tensor.numpy().copy()
        before = logits.copy()
        state = random.get_state().clone()
        if observer is not None:
            observer.observe('fixture', 23, step, context, 2, logits)
        assert np.array_equal(logits, before)
        assert torch.equal(state, random.get_state())
        token = int(torch.multinomial(torch.softmax(torch.from_numpy(logits), dim=0), 1, generator=random))
        selected.append(token); context.append(token); all_logits.append(logits)
    return selected, all_logits, random.get_state()


with tempfile.TemporaryDirectory() as temporary:
    observer = SourceLogitObserver(contract, 'suite', 'manifest', temporary)
    global_random = torch.get_rng_state().clone()
    before = sample()
    after = sample(observer)
    assert before[0] == after[0] and torch.equal(before[2], after[2])
    assert all(np.array_equal(a, b) for a, b in zip(before[1], after[1]))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    assert torch.equal(global_random, torch.get_rng_state())
    report = {'results': [{'id': 'fixture', 'seed': 23, 'inputIds': [3, 2], 'tokenIds': after[0],
                           'stopReason': 'max-tokens'}],
              'producerSha256': 'producer', 'evaluatorSha256': 'evaluator', 'packages': {},
              'arithmetic': 'synthetic F32', 'settings': {}, 'source': {}, 'deployed': {}}
    index = observer.write_index(Path(temporary) / 'index.json', report)
    assert index['observedPositions'] == 2 and index['requestedPositions'] == 3
    assert index['unobserved'] == [{'caseId': 'fixture', 'seed': 23, 'step': 7}]
    for row in index['observations']:
        raw = (Path(temporary) / row['logits']['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['logits']['sha256']
        assert np.array_equal(np.frombuffer(raw, dtype='<f4'), before[1][row['step']])
    for suite, manifest in [('wrong', 'manifest'), ('suite', 'wrong')]:
        try:
            SourceLogitObserver(contract, suite, manifest, temporary)
        except AssertionError:
            pass
        else:
            raise AssertionError('Mismatched observation identity accepted.')
    try:
        observer.observe('fixture', 23, 0, [3, 2], 2, before[1][0])
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicate observation accepted.')
print('source-logit-observation-test: pass')
