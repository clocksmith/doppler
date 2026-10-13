"""Pinned n-gram address behavior across chunks; cache port is an explicit spy."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
for entry in binding['files'] + binding['referenceImplementation']['files']:
    assert hashlib.sha256((ROOT / 'source' / entry['path']).read_bytes()).hexdigest() == entry['sha256']
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
config_tree = ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text())
config_class = next(n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
seed = next(n for n in config_class.body if isinstance(n, ast.AnnAssign) and n.target.id == 'seed')
config.setdefault('seed', ast.literal_eval(seed.value))
selected = {'_splitmix64', '_build_layer_multipliers', '_is_prime', '_find_nth_prime_after', 'Qwen4ExpTextNGramEmbedding'}
constants = {'_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2', '_PRIME_1'}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
for node in ast.parse((ROOT / 'source/modeling_qwen4_exp.py').read_text()).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected:
        assert not node.decorator_list
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
        nodes.append(node)

class AddressLookup(nn.Module):
    def __init__(self, count, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    def forward(self, indices):
        return indices.unsqueeze(-1)

class LookupProxy:
    def __getattr__(self, name):
        return AddressLookup if name == 'Embedding' else getattr(nn, name)

class TokenHistoryPort:
    """Explicit test port, not upstream Cache or Doppler state qualification."""
    def __init__(self, layer_index):
        self.layer_index = layer_index
        self.layers = {layer_index: SimpleNamespace(conv_states=[None, None, None])}
        self.updates = 0
    def has_previous_state(self, layer_idx, state_idx):
        assert layer_idx == self.layer_index and state_idx == 2
        return self.layers[self.layer_index].conv_states[2] is not None
    def update_conv_state(self, tokens, layer_idx, state_idx, conv_kernel_size):
        assert layer_idx == self.layer_index and state_idx == 2
        previous = self.layers[self.layer_index].conv_states[2]
        combined = tokens if previous is None else torch.cat([previous, tokens], dim=-1)
        state = combined[:, -conv_kernel_size:].clone()
        assert state.shape[-1] == conv_kernel_size
        self.layers[self.layer_index].conv_states[2] = state
        self.updates += 1
        return state

namespace = {'torch': torch, 'nn': LookupProxy(), 'math': math}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-ngram>', 'exec'), namespace)
torch.set_num_threads(2)
config = SimpleNamespace(**config)
operator = namespace['Qwen4ExpTextNGramEmbedding'](config, config.ple_embed_dim, 1, 0)
prior = json.loads((ROOT / 'reference-fixtures.json').read_text())
tokens = prior['ngram']['tokens'][0]
expected = torch.tensor(prior['ngram']['addresses']['values'], dtype=torch.long)
assert operator(torch.tensor([tokens]), None).equal(expected)
partitions = []
for mask in range(1 << (len(tokens)-1)):
    cuts = [0] + [i+1 for i in range(len(tokens)-1) if mask & (1 << i)] + [len(tokens)]
    cache = TokenHistoryPort(operator.layer_idx)
    actual = torch.cat([operator(torch.tensor([tokens[a:b]]), cache) for a,b in zip(cuts,cuts[1:])], dim=1)
    assert actual.equal(expected)
    assert cache.layers[operator.layer_idx].conv_states[2].tolist() == [tokens[-operator.context_len:]]
    partitions.append(cuts[1:-1])
# Interleave distinct histories on the same immutable operator instance.
other = [73, config.eos_token_id, 89, 90, 91, config.eos_token_id, 93, 94]
other_expected = operator(torch.tensor([other]), None)
a, b = TokenHistoryPort(operator.layer_idx), TokenHistoryPort(operator.layer_idx)
actual_a, actual_b = [], []
for i in range(len(tokens)):
    actual_a.append(operator(torch.tensor([[tokens[i]]]), a))
    actual_b.append(operator(torch.tensor([[other[i]]]), b))
assert torch.cat(actual_a,dim=1).equal(expected)
assert torch.cat(actual_b,dim=1).equal(other_expected)
assert a.layers[operator.layer_idx].conv_states[2].data_ptr() != b.layers[operator.layer_idx].conv_states[2].data_ptr()
report = {
    'schema':'doppler.qwen38-ngram-incremental-reference/v1',
    'checkpoint':binding['revision'], 'upstream':binding['referenceImplementation']['revision'],
    'torch':torch.__version__, 'surface':'CPU',
    'scope':'Unchanged upstream address computation; embedding lookup and cache ports are test spies',
    'tokens':tokens, 'otherTokens':other, 'contextTokens':operator.context_len,
    'chunkPartitionsChecked':len(partitions), 'allChunkPartitionsMatchIndependentAddresses':True,
    'interleavedHistoriesMatch':True, 'independentMutableHistory':True,
    'expectedAddressesSha256':prior['ngram']['addresses']['sha256'],
    'passed':True, 'upstreamCacheQualified':False, 'learnedEmbeddingQualified':False, 'dopplerQualified':False,
}
(ROOT/'ngram-incremental-reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
