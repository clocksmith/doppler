"""Offline harness checks; synthetic logits never establish model quality."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path('tools/lib').resolve()))
import source_task_quality as evaluator


class Tokenizer:
    tokens = {0: '', 1: 'prompt', 2: '<think>', 3: '</think>', 4: 'A', 5: '<eos>', 6: '😀'}

    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[3] if text == '</think>' else [1])

    def decode(self, tokens, skip_special_tokens):
        return ''.join(self.tokens[token] for token in tokens
                       if not skip_special_tokens or token not in [0, 1, 2, 3, 5])


class SourceTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'prompt'

    def encode(self, rendered, **kwargs):
        return [1]


class Model:
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def __call__(self, **kwargs):
        scores = torch.full((1, 1, 7), -1000.0)
        scores[0, 0, next(self.sequence)] = 1000.0
        return SimpleNamespace(logits=scores)


base = {'enableThinking': True, 'maxOutputCharacters': 64,
        'answerBoundary': {'text': '</think>', 'tokenId': 3, 'exactlyOne': True},
        'generation': {'temperature': 0, 'topK': 1, 'topP': 1, 'repetitionPenalty': 1.0,
                       'repetitionPenaltyWindow': 0, 'presencePenalty': 0, 'stopSequences': [],
                       'maxTokens': 8, 'maxSeqLen': 16},
        'cases': [{'id': 'synthetic-boundary', 'prompt': 'prompt', 'expected': 'A'}]}


def check(sequence, suite):
    report = {'results': []}
    with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate_tasks(Model(sequence), SimpleNamespace(pad_token_id=None, eos_token_id=5),
                                 Tokenizer(), SourceTokenizer(), suite, report, Path(root) / 'result.json')
    return report['results'][0]


evaluator.StoredHalfCache = lambda **kwargs: object()
valid = check([2, 4, 3, 4, 5], copy.deepcopy(base))
assert valid['passed'] and valid['answerText'] == 'A' and valid['rawDecodedText'] == '<think>A</think>A<eos>'
missing = check([2, 4, 5], copy.deepcopy(base))
assert not missing['passed'] and missing['stopReason'] == 'eos' and missing['answerBoundaryPositions'] == []
duplicate = check([3, 4, 3, 5], copy.deepcopy(base))
assert not duplicate['passed'] and len(duplicate['answerBoundaryPositions']) == 2
limited = copy.deepcopy(base)
limited['maxOutputCharacters'] = 1
characters = check([6, 3, 4, 5], limited)
assert characters['rawDecodedCharacters'] == 2 and characters['stopReason'] == 'output-character-limit'
assert characters['tokenIds'] == [6] and not characters['passed']
limited = copy.deepcopy(base)
limited['generation']['maxTokens'] = 3
tokens = check([2, 4, 4, 3, 5], limited)
assert tokens['stopReason'] == 'max-tokens' and tokens['tokenIds'] == [2, 4, 4] and not tokens['passed']
sampled = copy.deepcopy(base)
sampled['seeds'] = [17]
sampled['generation'].update({'temperature': 0.6, 'topK': 2, 'topP': 0.95, 'minP': 0})
first = check([2, 4, 3, 4, 5], sampled)
second = check([2, 4, 3, 4, 5], sampled)
assert first == second and first['passed'] and first['seed'] == 17
print(json.dumps({'syntheticHarnessChecks': 6, 'passed': True, 'modelQualification': False}))
