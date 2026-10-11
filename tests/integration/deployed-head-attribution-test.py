"""Synthetic controls for same-operand head attribution, not model qualification."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
from transformers import Qwen3Config

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from lib.deployed_head_attribution import probabilities, probe_deployed_head


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 6, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.arange(24).reshape(6, 4) / 100)

    def get_output_embeddings(self):
        return self.head

    def forward(self, **_kwargs):
        return SimpleNamespace(logits=self.head(torch.ones(1, 1, 4)))


class Tokenizer:
    def encode(self, text, **_kwargs):
        assert text == 'fixture'
        return SimpleNamespace(ids=[0])

    def decode(self, ids, **_kwargs):
        return str(ids[0])


model = Model()
settings = {'repetitionPenaltyWindow': 0, 'presencePenalty': 0, 'stopSequences': [], 'minP': 0,
            'repetitionPenalty': 1, 'temperature': .6, 'topK': 3, 'topP': .95}
logits = model().logits[0, -1].detach()
random = torch.Generator().manual_seed(17)
history, tokens = [0], []
for _ in range(3):
    token = int(torch.multinomial(probabilities(logits, history, settings, None), 1, generator=random).item())
    tokens.append(token); history.append(token)
report = {'settings': settings, 'source': {'fixture': True}, 'deployed': {'fixture': True}}
baseline = {**report, 'results': [{'id': 'fixture', 'seed': 17, 'tokenIds': tokens,
                                  'inputIds': [0], 'renderedPrompt': 'fixture'}]}
task = {'caseId': 'fixture', 'seed': 17, 'selectedStep': 2, 'selectedTokenId': tokens[-1], 'eosTokenIds': [0]}
config = Qwen3Config(num_hidden_layers=1, hidden_size=4, num_attention_heads=1,
                     num_key_value_heads=1, pad_token_id=None)
with tempfile.TemporaryDirectory() as temporary:
    output = Path(temporary) / 'result.json'
    same = dict(report)
    probe_deployed_head(model, model.head.weight.detach().clone(), config, Tokenizer(), baseline, task, same, output)
    assert same['allRetainedPrefixTokensMatch'] and same['deployedHeadReplayExact']
    assert not same['selectedTokenChanged'] and same['deployedHead'] == same['originalHead']
    assert json.loads(output.read_text())['sameRngPosition']
    alternate = torch.full_like(model.head.weight, -100); alternate[0] = 100
    changed = dict(report)
    probe_deployed_head(model, alternate, config, Tokenizer(), baseline, task, changed, output)
    assert changed['selectedTokenChanged'] and changed['sourceHeadSelectsEos']
    invalid = {**baseline, 'results': [{**baseline['results'][0], 'tokenIds': [0, *tokens[1:]]}]}
    try:
        probe_deployed_head(model, alternate, config, Tokenizer(), invalid, task, dict(report), output)
    except AssertionError as error:
        assert 'diverged at token 0' in str(error)
    else:
        raise AssertionError('Mismatched retained prefix must reject attribution.')
    assert not model.head._forward_pre_hooks, 'Observer releases on both success and failure.'
print('deployed-head-attribution: identical and changed heads, RNG replay, mismatch rejection and observer cleanup passed')
