"""CPU-only synthetic operator fixtures from the pinned upstream definitions.

No checkpoint weights, model execution, conversion or Doppler acceptance.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
for entry in binding['files'] + binding['referenceImplementation']['files']:
    assert hashlib.sha256((ROOT / 'source' / entry['path']).read_bytes()).hexdigest() == entry['sha256']
source = (ROOT / 'source/modeling_qwen4_exp.py').read_text()
config_source = ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text())
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
# Read upstream defaults rather than inventing values absent from checkpoint JSON.
config_class = next(n for n in config_source.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
for name in ['seed', 'norm_topk_prob']:
    node = next(n for n in config_class.body if isinstance(n, ast.AnnAssign) and n.target.id == name)
    config.setdefault(name, ast.literal_eval(node.value))
selected = {'Qwen4ExpTextRMSNorm', 'Qwen4ExpTextGatedResidual', 'Qwen4ExpTextTopKRouter',
            '_splitmix64', '_build_layer_multipliers', '_is_prime', '_find_nth_prime_after',
            'Qwen4ExpTextNGramEmbedding'}
constants = {'_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2', '_PRIME_1'}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
for node in ast.parse(source).body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected:
        assert not node.decorator_list
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
        nodes.append(node)
namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-upstream-operators>', 'exec'), namespace)
torch.set_num_threads(2)
torch.manual_seed(0)

def tensor_record(tensor):
    t = tensor.detach().cpu().contiguous()
    return {'shape': list(t.shape), 'dtype': str(t.dtype),
            'sha256': hashlib.sha256(t.numpy().tobytes()).hexdigest(), 'values': t.tolist()}

def fill_parameters(module):
    records = []
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(module.named_parameters()):
            values = ((torch.arange(parameter.numel(), dtype=torch.int64) * (ordinal + 1)) % 31 - 15).float() / 4096
            parameter.copy_(values.reshape(parameter.shape))
            records.append({'name':name, 'shape':list(parameter.shape), 'ordinal':ordinal,
                            'sha256':hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()})
    return records

cfg = SimpleNamespace(**config)
residual = namespace['Qwen4ExpTextGatedResidual'](cfg)
weights = fill_parameters(residual)
x = ((torch.arange(2 * cfg.hc_count * cfg.hidden_size) % 43 - 21).float() / 16).reshape(1,2,-1)
with torch.no_grad():
    mixed, unchanged, injection = residual(x)
assert unchanged.data_ptr() == x.data_ptr()
assert torch.isfinite(mixed).all() and torch.isfinite(injection).all()
try:
    residual(x[..., :-1])
    raise AssertionError('Malformed residual width accepted')
except ValueError:
    pass

# Capture only exact hash addresses, avoiding the 51B-parameter embedding allocation.
# The original upstream constructor/forward run unchanged; embedding lookup is a spy.
class IndexCapture(nn.Module):
    def __init__(self, count, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        self.geometry = [count, dim]
    def forward(self, indices):
        return indices.unsqueeze(-1)

class NNLookupProxy:
    def __getattr__(self, name):
        return IndexCapture if name == 'Embedding' else getattr(nn, name)
namespace['nn'] = NNLookupProxy()
ngram = namespace['Qwen4ExpTextNGramEmbedding'](cfg, cfg.ple_embed_dim, 1, 0)
ids = torch.tensor([[17, 42, cfg.eos_token_id, 101, 102, 103, cfg.eos_token_id, 248319]], dtype=torch.long)
with torch.no_grad():
    addresses = ngram(ids, None)
# Independent Python-integer calculation checks EOS reset and all head offsets.
expected = []
history = [cfg.eos_token_id] * (cfg.ngram_size - 1)
for token in ids[0].tolist():
    history.append(token)
    segments = history[:-1]
    prior_eos = max((i for i, v in enumerate(segments) if v == cfg.eos_token_id), default=-1)
    prior = [token] + [history[-1-shift] if len(history)-1-shift > prior_eos else cfg.eos_token_id for shift in range(1,cfg.ngram_size)]
    row = []
    for size in range(2, cfg.ngram_size+1):
        mixed_id = prior[0] * int(ngram.layer_multipliers[0])
        for i in range(1,size): mixed_id ^= prior[i] * int(ngram.layer_multipliers[i])
        for head in range((size-2)*cfg.heads_per_ngram,(size-1)*cfg.heads_per_ngram):
            row.append(mixed_id % ngram.head_vocab_sizes[head] + ngram.head_offsets[head])
    expected.append(row)
assert addresses.tolist() == [expected]
report = {'schema':'doppler.qwen38-reference-fixtures/v1', 'checkpoint':binding['revision'],
          'upstream':binding['referenceImplementation']['revision'], 'torch':torch.__version__,
          'surface':'CPU', 'threads':2, 'scope':'Synthetic operator controls; no checkpoint weights or Doppler execution',
          'sourceNodes':sorted(selected), 'defaults':{k:config[k] for k in ['seed','norm_topk_prob']},
          'gatedResidual':{'weightRecipe':'((arange * (ordinal + 1)) % 31 - 15) / 4096, float32',
            'weights':weights, 'inputRecipe':'((arange % 43 - 21) / 16).reshape(1,2,10240), float32',
            'inputSha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            'mixed':tensor_record(mixed), 'injection':tensor_record(injection), 'malformedWidthRejected':True},
          'ngram':{'mode':'upstream hashing with lookup spy, no learned embeddings', 'tokens':ids.tolist(),
            'multipliers':ngram.layer_multipliers.tolist(), 'headVocabularySizes':ngram.head_vocab_sizes,
            'headOffsets':ngram.head_offsets, 'embeddingGeometry':ngram.ngram_embedding.geometry,
            'addresses':tensor_record(addresses), 'independentIntegerCheck':True},
          'pending':['QSA indexer and sparse attention fixtures','Expert execution fixtures','Incremental PLE/cache fixtures',
                     'Learned-checkpoint independent reference','Conversion and installed distributed qualification']}
(ROOT / 'reference-fixtures.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'passed':True, 'checkpoint':binding['revision'], 'ngramTokens':ids.numel(),
                  'gatedResidualShape':list(mixed.shape), 'fullModelQualified':False}))
