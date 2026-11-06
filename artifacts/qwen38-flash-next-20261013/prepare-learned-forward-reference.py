"""Pinned upstream Flash-Next text forward on the learned checkpoint weights (CPU, one layer at a time).

Runs the unchanged upstream Qwen4ExpTextDecoderLayer stack (gated residuals, PLE with the real n-gram
table, gated delta net, QSA-indexed attention, sparse MoE) and the root hyper-connection mixer over
fixed token sequences, then the LM head. Weights are read from the pinned checkpoint files located by
the authenticated header receipt; each file must match acquisition-receipt.json by size (its SHA-256
was verified at acquisition). Layers are built and loaded one at a time, so peak memory is one
decoder layer. The forward mirrors Qwen4ExpTextModel.forward without a cache: embeddings replicated
over the streams, text positions expanded to the 3 mrope rows, an eager float causal mask for
indexed attention, and no linear-attention mask (unpadded input). The n-gram table rows are read
from the checkpoint by address instead of materializing the 95 GiB embedding.

Records per-layer stream samples, QSA selections and block-score margins at sampled queries, router
top-k and margins, gated-delta decay ranges, and per-position logits (top-k, argmax; full rows for
selected positions as .npy beside the JSON).
Optional --artifact <dir> reads stored F16/Q4_K weights with independent gguf dequantization,
while keeping the unchanged upstream forward. It preserves the source-only control separately.
Usage: python3 -P prepare-learned-forward-reference.py <source-dir> <out-dir> [--dtype float32|bfloat16]
       [--artifact <dir>] [--sequences short,long] [--kv-dtype f16]
"""
import ast
import hashlib
import json
import math
import struct
import sys
import time
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
DTYPE = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[sys.argv[sys.argv.index('--dtype') + 1] if '--dtype' in sys.argv else 'float32']
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACT = Path(sys.argv[sys.argv.index('--artifact') + 1]) if '--artifact' in sys.argv else None
KV_F16 = '--kv-dtype' in sys.argv
if KV_F16:
    assert sys.argv[sys.argv.index('--kv-dtype') + 1] == 'f16' and ARTIFACT
PRODUCER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
artifact_manifest = json.loads((ARTIFACT / 'manifest.json').read_text()) if ARTIFACT else None
if ARTIFACT:
    from gguf import quants, GGMLQuantizationType
    assert DTYPE == torch.float32, 'artifact control uses f32 arithmetic on independently decoded stored weights'
binding = json.loads((ROOT / 'source-binding.json').read_text())
for entry in binding['files'] + binding['referenceImplementation']['files']:
    assert hashlib.sha256((ROOT / 'source' / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
headers = json.loads((ROOT / 'checkpoint-headers.json').read_text())
acquired = {f['file']: f for f in json.loads((ROOT / 'acquisition-receipt.json').read_text())['files']}
assert headers['revision'] == binding['revision']
if artifact_manifest:
    assert artifact_manifest['artifactIdentity']['sourceRevision'] == binding['revision']
    assert artifact_manifest['moeConfig']['expertFormat'] == 'qwen3_5'
config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
config_class = next(n for n in ast.parse((ROOT / 'source/configuration_qwen4_exp.py').read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextConfig')
defaults = {}
for n in config_class.body:  # omitted source fields take the pinned class defaults (as Qwen4ExpTextConfig does)
    if isinstance(n, ast.AnnAssign) and n.value is not None:
        try:
            defaults[n.target.id] = ast.literal_eval(n.value)
        except ValueError:
            pass
for key, value in defaults.items():
    config.setdefault(key, value)
assert config['seed'] == 1234 and 'norm_topk_prob' in config
cfg = SimpleNamespace(**config)
cfg.ple_embed_dim = cfg.ple_embed_dim or cfg.hidden_size
cfg._attn_implementation = 'eager'
cfg._experts_implementation = 'eager'
model_source = (ROOT / 'source/modeling_qwen4_exp.py').read_text()
for anchor in ['hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)',
               'hidden_states = self.hyper_connection_mixer(hidden_states)',
               'position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)',
               '"indexed_attention": create_causal_mask(**mask_kwargs)']:
    assert anchor in model_source, anchor

# ---- Unchanged upstream definitions (decorators asserted, then removed as in the other harnesses) ----
def source_nodes(path, names, decorators=None, method_decorators=None, keep_methods=()):
    found = {}
    for node in ast.parse((ROOT / 'source' / path).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            assert [ast.unparse(d) for d in node.decorator_list] == (decorators or {}).get(node.name, []), node.name
            node.decorator_list = []
            for method, expected in (method_decorators or {}).get(node.name, {}).items():
                member = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == method)
                assert [ast.unparse(d) for d in member.decorator_list] == expected, method
                member.decorator_list = []
            found[node.name] = node
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            found[node.targets[0].id] = node
    assert set(found) == set(names), set(names) - set(found)
    return [found[name] for name in names]

def run(nodes, namespace, label):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), label, 'exec'), namespace)

class EagerOnlyRegistry:
    @staticmethod
    def get_interface(name, default):
        if name != 'eager':
            raise ValueError(f'unsupported implementation {name!r}')
        return default

row_width = cfg.ple_embed_dim // ((cfg.ngram_size - 1) * cfg.heads_per_ngram)
table = headers['ngramTable']
shard_rows = table['shardRows'][0]
row_bytes = table['rowWidth'] * 2
table_shards = {s['shard']: s for s in table['shards']}
handles = {}

def handle(name):
    if name not in handles:
        path = SOURCE / name
        assert path.stat().st_size == acquired[name]['bytes'], name
        handles[name] = open(path, 'rb')
    return handles[name]

rows_read = [0]
class TableRows(nn.Module):
    """Reads n-gram table rows from the checkpoint by address (header receipt spans); BF16 bytes kept exact."""
    def __init__(self, count, width):
        super().__init__()
        assert count == table['rows'] and width == row_width
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    def forward(self, indices):
        flat = indices.reshape(-1).tolist()
        out = np.empty((len(flat), row_width), dtype=np.uint16)
        for i, row in enumerate(flat):
            shard = table_shards[row // shard_rows]
            h = handle(shard['file'])
            h.seek(shard['fileByteBegin'] + (row % shard_rows) * row_bytes)
            out[i] = np.frombuffer(h.read(row_bytes), dtype=np.uint16)
        rows_read[0] += len(flat)
        values = torch.from_numpy((out.astype(np.uint32) << 16).view(np.float32))
        return values.reshape(*indices.shape, row_width).to(DTYPE)

class NNProxy:
    def __getattr__(self, name):
        return TableRows if name == 'Embedding' else getattr(nn, name)

activations = ast.parse((ROOT / 'source/activations.py').read_text())
act2cls = next(n.value for n in activations.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ACT2CLS')
mapping = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(act2cls.keys, act2cls.values)}
assert mapping['silu'] == 'SiLUActivation' and mapping['sigmoid'] == 'nn.Sigmoid'
namespace = {'torch': torch, 'nn': NNProxy(), 'F': F, 'math': math, 'Tensor': torch.Tensor, 'Cache': object,
             'TransformersKwargs': dict, 'Unpack': type('Unpack', (), {'__class_getitem__': classmethod(lambda cls, item: item)}),
             'GradientCheckpointingLayer': nn.Module, 'is_torchdynamo_exporting': lambda: False,
             'is_torchdynamo_compiling': lambda: False, 'ALL_ATTENTION_FUNCTIONS': EagerOnlyRegistry,
             'logger': SimpleNamespace(warning_once=lambda *a, **k: None), 'Callable': object, 'wraps': wraps}
run(source_nodes('activations.py', ['SiLUActivation'], {'SiLUActivation': ["use_kernel_forward_from_hub('SiLU')"]}), namespace, '<act>')
namespace['ACT2FN'] = {'silu': namespace['SiLUActivation'](), 'sigmoid': nn.Sigmoid()}
run([n for n in ast.parse((ROOT / 'source/modeling_rope_utils.py').read_text()).body
     if isinstance(n, ast.FunctionDef) and n.name == 'dynamic_rope_update'], namespace, '<rope-utils>')
names = ['Qwen4ExpTextRotaryEmbedding', 'Qwen4ExpTextRMSNorm', 'Qwen4ExpTextRMSNormGated', 'apply_mask_to_padding_states',
         'causal_conv1d_update', 'causal_conv1d_fn', 'l2norm', 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
         'Qwen4ExpTextGatedDeltaNet', 'rotate_half', 'apply_rotary_pos_emb', 'Qwen4ExpTextQSAIndexer', 'repeat_kv',
         'eager_attention_forward', 'Qwen4ExpTextAttention', 'Qwen4ExpTextMLP', 'Qwen4ExpTextExperts', 'Qwen4ExpTextTopKRouter',
         'Qwen4ExpTextSparseMoeBlock', 'Qwen4ExpTextGatedResidual', '_MASK64', '_SPLITMIX_GAMMA', '_SPLITMIX_M1', '_SPLITMIX_M2',
         '_PRIME_1', '_splitmix64', '_build_layer_multipliers', '_is_prime', '_find_nth_prime_after', 'Qwen4ExpTextNGramEmbedding',
         'Qwen4ExpTextPLELayer', 'Qwen4ExpTextDecoderLayer']
decorators = {
    'Qwen4ExpTextRMSNormGated': ["use_kernel_forward_from_hub('RMSNormGated')"],
    'causal_conv1d_update': ["use_kernel_func_from_hub_with_fallback('causal_conv1d_update', 'causal_conv1d')"],
    'causal_conv1d_fn': ["use_kernel_func_from_hub_with_fallback('causal_conv1d_fn', 'causal_conv1d')"],
    'torch_chunk_gated_delta_rule': ["use_kernel_func_from_hub_with_fallback('chunk_gated_delta_rule', 'fla')"],
    'torch_recurrent_gated_delta_rule': ["use_kernel_func_from_hub_with_fallback('fused_recurrent_gated_delta_rule', 'fla')"],
    'Qwen4ExpTextGatedDeltaNet': ['use_kernelized_func([torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update])'],
    'Qwen4ExpTextExperts': ['use_experts_implementation'],
}
run(source_nodes('modeling_qwen4_exp.py', names, decorators, {'Qwen4ExpTextGatedDeltaNet': {'forward': ["force_accelerate_hooks('conv1d')"]}}),
    namespace, '<pinned-qwen4-exp-text>')
if KV_F16:
    # Explicit storage-precision control: retain the upstream attention math,
    # rounding only post-RoPE K/V at the point a runtime F16 cache stores them.
    upstream_attention = namespace['eager_attention_forward']
    def stored_kv_attention(module, query, key, value, *args, **kwargs):
        return upstream_attention(module, query, key.to(torch.float16).float(),
                                  value.to(torch.float16).float(), *args, **kwargs)
    namespace['eager_attention_forward'] = stored_kv_attention
torch.set_num_threads(16)
assert torch.get_float32_matmul_precision() == 'highest'

# ---- Checkpoint tensors -------------------------------------------------------------------------
locations = {}
for name, info in headers['files'].items():
    for tensor, t in info['tensors'].items():
        locations[tensor] = (name, info['dataStart'], t)

def read_source_tensor(name):
    file, data_start, t = locations[name]
    begin, end = t['dataOffsets']
    h = handle(file)
    h.seek(data_start + begin)
    raw = h.read(end - begin)
    if t['dtype'] == 'BF16':
        bits = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
        return bits.reshape(t['shape']).to(DTYPE)
    assert t['dtype'] == 'I64', t['dtype']
    return torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(t['shape']).clone()

def read_tensor(name):
    if not artifact_manifest:
        return read_source_tensor(name)
    # Integer checkpoint buffers are source-owned contracts, not converted weights.
    if locations[name][2]['dtype'] == 'I64':
        return read_source_tensor(name)
    t = artifact_manifest['tensors'][name]
    assert t.get('layout', 'row') == 'row', name
    spans = t.get('spans') or [{'shardIndex': t['shard'], 'offset': t['offset'], 'size': t['size']}]
    raw = bytearray(t['size'])
    cursor = 0
    for span in spans:
        shard = artifact_manifest['shards'][span['shardIndex']]
        with open(ARTIFACT / shard['filename'], 'rb') as f:
            f.seek(span['offset'])
            chunk = f.read(span['size'])
        assert len(chunk) == span['size'], name
        raw[cursor:cursor + len(chunk)] = chunk
        cursor += len(chunk)
    assert cursor == t['size'], name
    if t['dtype'] == 'F16':
        values = np.frombuffer(raw, dtype='<f2').astype(np.float32)
        return torch.from_numpy(values.reshape(t['shape'])).to(DTYPE)
    assert t['dtype'] in ('Q4_K', 'Q4_K_M'), (name, t['dtype'])
    width = t['shape'][-1]
    padded = ((width + 255) // 256) * 256
    rows = math.prod(t['shape'][:-1])
    assert len(raw) == rows * (padded // 256) * 144, name
    encoded = np.frombuffer(raw, dtype=np.uint8).reshape(rows, -1)
    values = quants.dequantize(encoded, GGMLQuantizationType.Q4_K)[:, :width].copy()
    return torch.from_numpy(values.reshape(t['shape'])).to(DTYPE)

def load(module, prefix):
    state = {}
    for key, value in module.state_dict().items():
        if 'ngram_embedding.weight' in key:
            continue
        tensor = read_tensor(prefix + key)
        assert tuple(tensor.shape) == tuple(value.shape), key
        if value.dtype == torch.int64:
            assert torch.equal(tensor, value), f'{key}: checkpoint buffer differs from the upstream derivation'
        state[key] = tensor
    stray = [n for n in locations if n.startswith(prefix) and n[len(prefix):] not in module.state_dict()
             and 'ngram_embedding.shard_' not in n]
    assert not stray, stray[:5]
    module.load_state_dict(state, strict=False, assign=True)
    missing = [k for k, v in module.state_dict().items() if k not in state and 'ngram_embedding.weight' not in k]
    assert not missing, missing[:5]

# ---- Inputs ----------------------------------------------------------------------------------
sequences = json.loads((ROOT / 'learned-forward-inputs.json').read_text())['sequences']
if '--sequences' in sys.argv:
    selected = sys.argv[sys.argv.index('--sequences') + 1].split(',')
    assert selected and len(set(selected)) == len(selected)
    assert all(any(s['id'] == name for s in sequences) for name in selected)
    sequences = [s for s in sequences if s['id'] in selected]
assert all(max(s['tokens']) < cfg.vocab_size for s in sequences)

# ---- Instrumentation -------------------------------------------------------------------------
SAMPLE_STRIDE = 16
records = {s['id']: {'layers': {}, 'qsa': {}, 'router': {}, 'decay': {}} for s in sequences}
current = {'seq': None, 'layer': None}

def sample_positions(T):
    picks = {0, 1, 2, 7, 31, 63, 255, 1023, 2050, 2051, 2052, 2053, 2100, 2200, T - 2, T - 1}
    return sorted(p for p in picks if 0 <= p < T)

def router_hook(module, inputs, output):
    logits = F.linear(inputs[0].float().reshape(-1, inputs[0].shape[-1]), module.weight.float())
    top = logits.topk(cfg.num_experts_per_tok + 1, dim=-1)
    rec = records[current['seq']]['router'].setdefault(str(current['layer']), {})
    positions = sample_positions(logits.shape[0])
    rec['positions'] = positions
    rec['topExperts'] = top.indices[positions, :cfg.num_experts_per_tok].tolist()
    rec['marginKthToNext'] = (top.values[:, cfg.num_experts_per_tok - 1] - top.values[:, cfg.num_experts_per_tok]).tolist()
    rec['minMargin'] = float(min(rec['marginKthToNext']))

def decay_hook(gdn):
    def hook(module, inputs, output):
        g = -gdn.A_log.float().exp() * F.softplus(output.float() + gdn.dt_bias.float())
        records[current['seq']]['decay'][str(current['layer'])] = {'min': float(g.min()), 'max': float(g.max())}
    return hook

def indexer_hook(module, inputs, output):
    hidden, (full_cos, full_sin), mask, _ = inputs
    T = hidden.shape[1]
    sparse = [q for q in sample_positions(T) if q + 1 > module.token_budget + module.compress_ratio - 1]
    if not sparse:
        return
    qk = module.index_qk_proj(hidden)
    q, token_k = torch.split(qk, [module.index_n_heads * module.index_head_dim, module.index_kv_heads * module.index_head_dim], dim=-1)
    q = module.q_layernorm(q.reshape(1, T, -1, module.index_head_dim))
    q = namespace['apply_rotary_pos_emb'](q, cos=full_cos[:, -T:, :], sin=full_sin[:, -T:, :], unsqueeze_dim=2)
    raw = token_k.reshape(1, T, module.index_head_dim)
    rec = records[current['seq']]['qsa'].setdefault(str(current['layer']), {})
    for query in sparse:
        blocks = (query + 1) // module.compress_ratio
        idx = torch.arange(blocks * module.compress_ratio).view(blocks, module.compress_ratio)
        pooled = module.k_layernorm(raw[0].index_select(0, idx.flatten()).view(blocks, module.compress_ratio, -1).float().mean(1).to(raw.dtype))
        keys = namespace['apply_rotary_pos_emb'](pooled.unsqueeze(1), cos=full_cos[0].index_select(0, idx[:, 0]),
                                                 sin=full_sin[0].index_select(0, idx[:, 0])).squeeze(1)
        scores = (torch.relu(q[0, query].float() @ keys.float().T).sum(0) / math.sqrt(module.index_head_dim))
        ordered = scores.sort(descending=True)
        visible = (output[0, 0, query] == 0) if output.is_floating_point() else output[0, 0, query]
        selected = torch.nonzero(visible).flatten().tolist()
        rec[str(query)] = {'selectedTokens': len(selected), 'selectedSha256': hashlib.sha256(np.array(selected, dtype=np.int32).tobytes()).hexdigest(),
                           'selectedBlocks': sorted(int(b) for b in set(t // module.compress_ratio for t in selected if t < blocks * module.compress_ratio)),
                           'blockMargin': float(ordered.values[module.block_topk - 1] - ordered.values[module.block_topk]) if blocks > module.block_topk else None,
                           'completeBlocks': blocks}

# ---- Forward ---------------------------------------------------------------------------------
embed = read_tensor('model.language_model.embed_tokens.weight')
rotary = namespace['Qwen4ExpTextRotaryEmbedding'](cfg)
states = {}
masks, positional = {}, {}
for s in sequences:
    ids = torch.tensor([s['tokens']])
    T = ids.shape[1]
    hidden = embed[ids[0]].unsqueeze(0).to(DTYPE)
    position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)[1:]
    positional[s['id']] = rotary(hidden, position_ids)
    masks[s['id']] = torch.full((T, T), torch.finfo(DTYPE).min, dtype=DTYPE).triu(1)[None, None]
    states[s['id']] = hidden.repeat(1, 1, cfg.hc_count)
del embed
timings = []
with torch.no_grad():
    for layer_idx in range(cfg.num_hidden_layers):
        started = time.time()
        layer = namespace['Qwen4ExpTextDecoderLayer'](cfg, layer_idx).eval()
        load(layer, f'model.language_model.layers.{layer_idx}.')
        layer.to(DTYPE)
        hooks = [layer.mlp.gate.register_forward_hook(router_hook)]
        if layer.layer_type == 'linear_attention':
            hooks.append(layer.linear_attn.in_proj_a.register_forward_hook(decay_hook(layer.linear_attn)))
        else:
            hooks.append(layer.self_attn.indexer.register_forward_hook(indexer_hook))
        for s in sequences:
            current.update(seq=s['id'], layer=layer_idx)
            T = len(s['tokens'])
            out = layer(states[s['id']], position_embeddings=positional[s['id']], attention_mask=masks[s['id']],
                        conv_mask=None, past_key_values=None, ple_input_ids=torch.tensor([s['tokens']]))
            states[s['id']] = out
            positions = sample_positions(T)
            records[s['id']]['layers'][str(layer_idx)] = {
                'rms': out[0].float().pow(2).mean(-1).sqrt()[positions].tolist(),
                'samples': out[0, positions, ::SAMPLE_STRIDE].float().tolist()}
        for h in hooks:
            h.remove()
        del layer
        timings.append(round(time.time() - started, 1))
        print(json.dumps({'layer': layer_idx, 'seconds': timings[-1], 'tableRows': rows_read[0]}), flush=True)
    mixer = namespace['Qwen4ExpTextGatedResidual'](cfg, use_combine=False).eval()
    load(mixer, 'model.language_model.hyper_connection_mixer.')
    mixer.to(DTYPE)
    head = read_tensor('lm_head.weight').float()
    summary = {}
    for s in sequences:
        mixed = mixer(states[s['id']])
        T = mixed.shape[1]
        positions = sample_positions(T)
        logits = mixed[0].float() @ head.T
        top = logits.topk(20, dim=-1)
        full_rows = {p: logits[p].numpy() for p in positions}
        suffix = ('-artifact' if ARTIFACT else '') + ('-kvf16' if KV_F16 else '')
        npy = OUT / f"{s['id']}-logits-{DTYPE.__repr__().split('.')[-1]}{suffix}.npy"
        np.save(npy, np.stack([full_rows[p] for p in positions]))
        summary[s['id']] = {
            'tokens': len(s['tokens']), 'samplePositions': positions,
            'mixedSamples': mixed[0, positions, ::SAMPLE_STRIDE // 4].float().tolist(),
            'argmax': logits.argmax(-1).tolist(), 'top20Ids': top.indices.tolist(), 'top20Logits': top.values.tolist(),
            'top1Margin': (top.values[:, 0] - top.values[:, 1]).tolist(),
            'logitsFile': npy.name, 'logitsSha256': hashlib.sha256(npy.read_bytes()).hexdigest(), **records[s['id']]}

result = {
    'schema': 'doppler.qwen38-learned-forward-reference/v1',
    'checkpoint': binding['revision'], 'upstream': binding['referenceImplementation']['revision'],
    'torch': torch.__version__, 'surface': 'CPU', 'dtype': str(DTYPE).split('.')[-1], 'threads': torch.get_num_threads(),
    'scope': 'Unchanged upstream text decoder stack, root mixer and LM head on the learned checkpoint weights, one cache-free '
             'full-sequence forward per input; vision and MTP excluded; no Doppler execution.',
    'inputs': 'learned-forward-inputs.json', 'sampleStride': SAMPLE_STRIDE, 'tableRowsRead': rows_read[0],
    'layerSeconds': timings, 'sequences': summary,
    'producerSha256': PRODUCER_SHA256,
}
if ARTIFACT:
    result['scope'] = 'Unchanged upstream cache-free text forward with stored F16 and independently gguf-dequantized Q4_K weights; source n-gram rows and integer buffers; no Doppler execution.'
    result['artifactManifestSha256'] = hashlib.sha256((ARTIFACT / 'manifest.json').read_bytes()).hexdigest()
    result['quantizationControl'] = {'decoder': 'gguf.quants.dequantize', 'type': 'Q4_K', 'rowPadding': 256}
    if KV_F16:
        result['scope'] += ' Post-RoPE keys and values round through F16 before unchanged upstream attention, matching the declared runtime KV storage.'
        result['kvStorageDtype'] = 'f16'
suffix = ('-artifact' if ARTIFACT else '') + ('-kvf16' if KV_F16 else '')
target = OUT / f"learned-forward-reference-{result['dtype']}{suffix}.json"
target.write_text(json.dumps(result, separators=(',', ':')) + '\n')
print(json.dumps({'written': str(target), 'bytes': target.stat().st_size,
                  'argmaxHead': {k: v['argmax'][-4:] for k, v in summary.items()}}))
