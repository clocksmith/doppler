#!/usr/bin/env python3
"""Route 3 requester: Qwen 3.5 with every projection evaluated on 2-party additive shares.

The requester holds the embedding, the norm weights, the DeltaNet convolution, A_log and dt_bias, and its KV and
recurrent state. Each projection group (input projections, output projection, gate+up, down, LM head) is encoded in
Z_(2^64) fixed point, split into two fresh shares, and sent to the stage's two operators, which return W~ x_b. The
requester reconstructs and evaluates every nonlinear function in the clear. The model equations are Transformers'
Qwen 3.5 (as in tools/qwen35-deployed-reference.py), with the deployed F16 KV storage.

  requester.py plan --model DIR --cuts 12 --out PLAN.json [--layers 1]
  requester.py layer --model DIR --plan PLAN.json --operators OPS.json --out REPORT.json
  requester.py generate --model DIR --plan PLAN.json --operators OPS.json --prompts PROMPTS.json --max-tokens 32 --out REPORT.json
  requester.py control --model DIR --prompts PROMPTS.json --max-tokens 32 --out REPORT.json   (plaintext linears)
  requester.py run --model DIR --plan PLAN.json --operators OPS.json --request REQUEST.json --out REPORT.json
      One request for a supervising process: a JSON event per line on stdout (ready, token, end or failed); a line
      `stop` on stdin stops the answer after the token in progress. REQUEST.json: {"messages": [...], "maxTokens": N}.

OPS.json: {"token": "...", "stages": [[{"host": h, "port": p}, {"host": h, "port": p}], ...]} (share 0, share 1).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import resource
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAGIC = 0x31535352
PLAN_SCHEMA = 'doppler.secret-shared-plan/v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def tensor_spans(manifest, entry):
    spans = entry.get('spans') or [{'shardIndex': entry['shard'], 'offset': entry['offset'], 'size': entry['size']}]
    return [{'file': manifest['shards'][s['shardIndex']]['filename'], 'offset': s['offset'], 'size': s['size']} for s in spans]


def read_tensor_bytes(root, manifest, entry):
    out = bytearray()
    for span in tensor_spans(manifest, entry):
        with open(Path(root) / span['file'], 'rb') as f:
            f.seek(span['offset'])
            out += f.read(span['size'])
    assert len(out) == entry['size']
    return bytes(out)


# ---------------------------------------------------------------- plan: which projections each stage's pair holds

def layer_groups(manifest, layer):
    p = f'model.language_model.layers.{layer}.'
    kind = manifest['inference']['layerPattern']['layerTypes'][layer]
    if kind == 'linear_attention':
        a = p + 'linear_attn.'
        groups = [('in', [a + 'in_proj_qkv.weight', a + 'in_proj_z.weight', a + 'in_proj_b.weight', a + 'in_proj_a.weight']),
                  ('out', [a + 'out_proj.weight'])]
    else:
        a = p + 'self_attn.'
        groups = [('in', [a + 'q_proj.weight', a + 'k_proj.weight', a + 'v_proj.weight']), ('out', [a + 'o_proj.weight'])]
    groups += [('up', [p + 'mlp.gate_proj.weight', p + 'mlp.up_proj.weight']), ('down', [p + 'mlp.down_proj.weight'])]
    return [(f'L{layer}.{name}', names) for name, names in groups]


def build_plan(root, cuts, fx, fw, layers=None):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    total = layers or manifest['architecture']['numLayers']
    bounds = [0, *cuts, total]
    stages = []
    for index in range(len(bounds) - 1):
        groups = []
        for layer in range(bounds[index], bounds[index + 1]):
            groups += layer_groups(manifest, layer)
        if index == len(bounds) - 2 and layers is None:
            groups.append(('head', ['model.language_model.lm_head.weight']))
        stage = {'index': index, 'layers': [bounds[index], bounds[index + 1] - 1], 'groups': []}
        for gid, names in groups:
            tensors = []
            for name in names:
                entry = manifest['tensors'][name]
                tensors.append({'name': name, 'dtype': entry['dtype'], 'shape': entry['shape'],
                                'spans': tensor_spans(manifest, entry),
                                'sha256': sha256_bytes(read_tensor_bytes(root, manifest, entry))})
            stage['groups'].append({'id': gid, 'in': tensors[0]['shape'][1], 'tensors': tensors})
        stages.append(stage)
    return {'schema': PLAN_SCHEMA, 'modelId': manifest['modelId'],
            'manifestSha256': sha256_bytes((root / 'manifest.json').read_bytes()), 'fx': fx, 'fw': fw,
            'stages': stages}


# ---------------------------------------------------------------- wire protocol and the operator pair

class Operator:
    def __init__(self, host, port, token):
        self.sock = socket.create_connection((host, port), timeout=600)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        self.file = self.sock.makefile('rb', buffering=1 << 20)
        self.sent = self.received = 0
        self.hello = self.call({'op': 'hello', 'token': token})[0]
        if 'error' in self.hello:
            raise RuntimeError(f'operator refused: {self.hello}')

    def call(self, header, payload=b''):
        js = json.dumps(header).encode()
        frame = struct.pack('<II', MAGIC, len(js)) + js + struct.pack('<Q', len(payload))
        self.sock.sendall(frame + payload if len(payload) < (1 << 16) else frame)
        if len(payload) >= (1 << 16):
            self.sock.sendall(payload)
        self.sent += len(frame) + len(payload)
        magic, size = struct.unpack('<II', self._read(8))
        assert magic == MAGIC
        head = json.loads(self._read(size))
        (plen,) = struct.unpack('<Q', self._read(8))
        body = self._read(plen)
        self.received += 16 + size + plen
        if 'error' in head:
            raise RuntimeError(f'operator error: {head}')
        return head, body

    def _read(self, n):
        data = self.file.read(n)
        if len(data) != n:
            raise ConnectionError('operator closed the connection')
        return data

    def close(self, op='close'):
        try:
            head, _ = self.call({'op': op})
            return head
        finally:
            self.sock.close()


def prg(seed, counter, size):
    return hashlib.shake_256(seed + struct.pack('<Q', counter)).digest(size)


class SharedCluster:
    """Encodes, shares, dispatches and reconstructs each projection group; records per-round cost."""

    def __init__(self, plan, operators, seeded=True):
        self.fx, self.fw = plan['fx'], plan['fw']
        self.where = {}
        for stage in plan['stages']:
            for group in stage['groups']:
                self.where[group['id']] = (stage['index'], group)
        token = operators['token']
        self.pairs = [[Operator(o['host'], o['port'], token) for o in pair] for pair in operators['stages']]
        plan_sha = sha256_bytes(json.dumps(plan).encode())
        for index, pair in enumerate(self.pairs):
            for share, op in enumerate(pair):
                h = op.hello
                assert h['stage'] == index and h['share'] == share and h['fx'] == self.fx and h['fw'] == self.fw, h
        self.seeded = seeded
        self.seeds = []
        for pair in self.pairs:
            seed = os.urandom(32)
            if seeded:
                pair[0].call({'op': 'seed', 'seed': seed.hex()})
            self.seeds.append(seed)
        self.counter = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.rounds = []
        self.cache = {}
        self.step = None  # the requester's current step, sent with every round so operators can account per token

    def run_group(self, gid, x):
        """x: float tensor [n, in]. Returns float32 [n, out] = W x, computed only on shares."""
        stage, group = self.where[gid]
        pair = self.pairs[stage]
        n = x.shape[0]
        t0 = time.perf_counter()
        xf = x.detach().to(torch.float64).numpy()
        enc = np.rint(np.ldexp(xf, self.fx))
        assert np.abs(enc).max() < 2.0 ** 62, 'input outside the fixed-point range'
        enc = enc.astype(np.int64).view(np.uint64)
        self.counter += 1
        counter = self.counter
        size = n * group['in'] * 8
        if self.seeded:
            x0 = np.frombuffer(prg(self.seeds[stage], counter, size), dtype='<u8').reshape(enc.shape)
        else:
            x0 = np.frombuffer(os.urandom(size), dtype='<u8').reshape(enc.shape)
        x1 = enc - x0  # uint64 wraps mod 2^64
        # Freshness evidence: share A is PRG(seed, counter) with a never-repeated counter; share B's digest is kept so
        # the receipt can show that no share vector was ever sent twice.
        share_b_digest = hashlib.sha256(x1.tobytes()).hexdigest()[:32]
        header = {'op': 'linear', 'group': gid, 'n': n, 'round': counter, **({'step': self.step} if self.step is not None else {})}
        t1 = time.perf_counter()
        sent_before = [op.sent for op in pair]
        recv_before = [op.received for op in pair]
        jobs = [self.pool.submit(pair[0].call, {**header, 'seeded': self.seeded}, b'' if self.seeded else x0.tobytes()),
                self.pool.submit(pair[1].call, {**header, 'seeded': False}, x1.tobytes())]
        (h0, y0), (h1, y1) = [job.result() for job in jobs]
        t2 = time.perf_counter()
        y = (np.frombuffer(y0, dtype='<u8') + np.frombuffer(y1, dtype='<u8')).view(np.int64)
        out = np.ldexp(y.astype(np.float64), -(self.fx + self.fw)).reshape(n, -1)
        assert np.abs(out).max() < 2.0 ** (62 - self.fx - self.fw), 'output outside the fixed-point range'
        t3 = time.perf_counter()
        self.rounds.append({'group': gid, 'stage': stage, 'n': n, 'counter': counter, 'shareBSha256': share_b_digest, 'encodeMs': (t1 - t0) * 1e3, 'wireMs': (t2 - t1) * 1e3,
                            'decodeMs': (t3 - t2) * 1e3, 'operatorMs': [h0['computeNs'] / 1e6, h1['computeNs'] / 1e6],
                            'up': [pair[i].sent - sent_before[i] for i in range(2)],
                            'down': [pair[i].received - recv_before[i] for i in range(2)]})
        return torch.from_numpy(out.astype(np.float32))

    def end_step(self, step):
        """Tells every operator that `step` ended, so each signs it before its output is acknowledged."""
        ops = [op for pair in self.pairs for op in pair]
        list(concurrent.futures.ThreadPoolExecutor(max_workers=len(ops)).map(lambda op: op.call({'op': 'step', 'step': step}), ops))

    def close(self, op='close'):
        return [[o.close(op) for o in pair] for pair in self.pairs]


class SharedLinear(nn.Module):
    """A projection whose weights live only at the operators. Members of one group share a single round."""

    def __init__(self, cluster, gid, start, end):
        super().__init__()
        self.cluster, self.gid, self.start, self.end = cluster, gid, start, end

    def forward(self, x):
        cached = self.cluster.cache.get(self.gid)
        if cached is None or cached[0] is not x:
            flat = x.reshape(-1, x.shape[-1])
            cached = (x, self.cluster.run_group(self.gid, flat))
            self.cluster.cache[self.gid] = cached
        out = cached[1][:, self.start:self.end]
        return out.reshape(*x.shape[:-1], self.end - self.start)


# ---------------------------------------------------------------- model

def load_model(root, source_config, plan=None, cluster=None, layers=None):
    """Qwen 3.5 on CPU in float32. With a cluster, projections are SharedLinear and their weights are never loaded."""
    from gguf import GGMLQuantizationType, dequantize
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    source = read_json(source_config)['text_config']
    if layers:
        source = {**source, 'num_hidden_layers': layers, 'layer_types': source['layer_types'][:layers]}
    config = Qwen3_5TextConfig(**source)
    config.tie_word_embeddings = False  # the deployed LM head is quantized separately
    config.dtype = 'float32'
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device='cpu')
    shared = {}
    if cluster is not None:
        for stage in plan['stages']:
            for group in stage['groups']:
                start = 0
                for t in group['tensors']:
                    rows = t['shape'][0]
                    shared[t['name']] = (group['id'], start, start + rows)
                    start += rows
    wanted = model.state_dict()
    weights = {}
    for name, entry in manifest['tensors'].items():
        if not name.startswith('model.language_model.'):
            continue
        key = name.replace('model.language_model.', 'model.', 1)
        if key == 'model.lm_head.weight':
            key = 'lm_head.weight'
        if key not in wanted:
            continue  # layers beyond a truncated model
        if name in shared:
            continue
        data = read_tensor_bytes(root, manifest, entry)
        if entry['dtype'] == 'Q4_K_M':
            array = dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K)
        elif key == 'model.embed_tokens.weight' and cluster is not None:
            # The requester keeps the embedding in its stored F16 form and widens only the rows it looks up.
            embedding_f16 = torch.from_numpy(np.frombuffer(data, dtype='<f2').copy().reshape(wanted[key].shape))
            del data
            continue
        else:
            array = np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4').astype(np.float32)
        weights[key] = torch.from_numpy(array.copy().reshape(wanted[key].shape))
        del data, array
    if cluster is not None:
        for name, (gid, start, end) in shared.items():
            key = name.replace('model.language_model.', 'model.', 1).removesuffix('.weight')
            if key == 'model.lm_head':
                key = 'lm_head'
            parent_name, _, attr = key.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, SharedLinear(cluster, gid, start, end))
    missing = [k for k in wanted if k not in weights and not any(
        k.removesuffix('.weight') == n.replace('model.language_model.', 'model.', 1).removesuffix('.weight')
        .replace('model.lm_head', 'lm_head') for n in shared)]
    if cluster is not None:
        missing = [k for k in missing if k != 'model.embed_tokens.weight']
    if layers:
        missing = [k for k in missing if k != 'lm_head.weight']
    assert not missing, missing[:5]
    if cluster is not None:
        model.model.embed_tokens = HalfEmbedding(embedding_f16)
    model.load_state_dict(weights, strict=False, assign=True)
    held = {key: int(value.numel() * value.element_size()) for key, value in model.state_dict().items() if value.device.type != 'meta'}
    model.requester_holds = {'tensors': len(held), 'bytes': sum(held.values()),
                             'embeddingBytes': held.get('model.embed_tokens.weight', 0),
                             'projectionBytes': sum(v for k, v in held.items() if k.endswith('proj.weight') or 'proj_' in k or k == 'lm_head.weight')}
    if layers and 'lm_head.weight' not in weights and cluster is None:
        model.lm_head = nn.Identity()
    model.eval()
    return model, config, manifest


class HalfEmbedding(nn.Module):
    """Embedding lookup over the stored F16 table; rows are widened to F32 after the lookup, as Doppler's F16 embedding."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, ids):
        return self.weight[ids].float()


def half_cache(config):
    """The deployed F16 KV storage with F32 attention arithmetic (as in qwen35-deployed-reference.py)."""
    try:  # transformers >= 5.6 routes Qwen 3.5 through the generic cache; 5.3 has its own hybrid cache class
        from transformers.models.qwen3_5.modeling_qwen3_5 import DynamicCache as Base
    except ImportError:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache as Base

    class StoredHalfCache(Base):
        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            keys, values = super().update(key_states.half(), value_states.half(), layer_idx, *args, **kwargs)
            return keys.float(), values.float()
    return StoredHalfCache(config=config)


def tokenize_prompts(root, prompts):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(root) / 'tokenizer.json'))
    rows = []
    for prompt in prompts:
        text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        rows.append({'prompt': prompt, 'text': text, 'ids': tokenizer.encode(text, add_special_tokens=False).ids})
    return tokenizer, rows


def greedy(model, config, manifest, ids, max_tokens, cluster=None, forced=None, on_token=None, should_stop=None):
    """Greedy decode (or teacher-forced on `forced`), returning tokens, per-step logits summaries and timings."""
    cache = half_cache(config)
    if forced is not None:
        max_tokens = min(max_tokens, len(forced))
    tokens, steps, logits_rows = [], [], []
    inputs = list(ids)
    eos = set(manifest['eos_token_id'] if isinstance(manifest['eos_token_id'], list) else [manifest['eos_token_id']])
    for step in range(max_tokens):
        if cluster:
            cluster.cache.clear()
            cluster.step = step
            before = len(cluster.rounds)
        t0, c0 = time.perf_counter(), time.process_time()
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([inputs]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1].float().numpy().astype(np.float64)
        wall = time.perf_counter() - t0
        top = np.argsort(-logits)[:2]
        token = int(top[0])
        row = {'step': step, 'columns': len(inputs), 'wallMs': wall * 1e3, 'cpuMs': (time.process_time() - c0) * 1e3, 'token': token,
               'margin': float(logits[top[0]] - logits[top[1]])}
        if cluster:
            rounds = cluster.rounds[before:]
            row.update({'rounds': len(rounds), 'up': sum(sum(r['up']) for r in rounds),
                        'down': sum(sum(r['down']) for r in rounds),
                        'wireMs': sum(r['wireMs'] for r in rounds),
                        'operatorMs': sum(max(r['operatorMs']) for r in rounds)})
        steps.append(row)
        logits_rows.append(logits.astype(np.float32))
        tokens.append(token)
        nxt = forced[step] if forced is not None else token
        done = forced is None and (token in eos or step + 1 == max_tokens)
        if cluster:
            cluster.end_step(step)
        if on_token:
            on_token(step, token, row, done)
        if forced is None and token in eos:
            break
        if should_stop and should_stop():
            row['stoppedAfter'] = True
            break
        inputs = [nxt]
    return tokens, steps, logits_rows


def peak_rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == 'darwin' else r * 1024


# ---------------------------------------------------------------- commands

def cmd_plan(args):
    cuts = [int(c) for c in args.cuts.split(',')] if args.cuts else []
    plan = build_plan(args.model, cuts, args.fx, args.fw, layers=args.layers)
    Path(args.out).write_text(json.dumps(plan, indent=1) + '\n')
    print(json.dumps({'stages': [{'index': s['index'], 'layers': s['layers'], 'groups': len(s['groups'])} for s in plan['stages']]}))


def cmd_layer(args):
    """C1: one layer on shares vs float64 W x per group, and vs the same layer with plaintext linears."""
    from gguf import GGMLQuantizationType, dequantize
    torch.set_num_threads(args.threads)
    plan = read_json(args.plan)
    layers = plan['stages'][-1]['layers'][1] + 1
    cluster = SharedCluster(plan, read_json(args.operators), seeded=not args.explicit)
    shared, config, manifest = load_model(args.model, args.source_config, plan, cluster, layers=layers)
    plain, _, _ = load_model(args.model, args.source_config, layers=layers)
    _, rows = tokenize_prompts(args.model, [args.prompt])
    ids = torch.tensor([rows[0]['ids']])
    groups = {}
    original = cluster.run_group

    def checked(gid, x):
        y = original(gid, x)
        _, group = cluster.where[gid]
        ref = []
        for t in group['tensors']:
            entry = manifest['tensors'][t['name']]
            data = read_tensor_bytes(args.model, manifest, entry)
            w = (dequantize(np.frombuffer(data, dtype=np.uint8).copy(), GGMLQuantizationType.Q4_K) if entry['dtype'] == 'Q4_K_M'
                 else np.frombuffer(data, dtype='<f2' if entry['dtype'] == 'F16' else '<f4')).astype(np.float64).reshape(t['shape'])
            ref.append(x.detach().to(torch.float64).numpy() @ w.T)
        ref = np.concatenate(ref, axis=1)
        err = np.abs(y.numpy().astype(np.float64) - ref)
        groups[gid] = {'n': int(x.shape[0]), 'maxAbs': float(err.max()), 'scale': float(np.abs(ref).max()),
                       'relative': float(err.max() / max(np.abs(ref).max(), 1e-30))}
        return y
    cluster.run_group = checked
    with torch.inference_mode():
        a = shared.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        b = plain.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    stats = cluster.close()
    err = (a - b).abs()
    report = {'schema': 'doppler.secret-shared-layer-check/v1', 'prompt': args.prompt, 'tokens': len(rows[0]['ids']),
              'layers': layers, 'groups': groups, 'layerMaxAbs': float(err.max()), 'layerScale': float(b.abs().max()),
              'note': 'last_hidden_state includes the final norm; both sides apply the same final norm',
              'rounds': cluster.rounds, 'operatorStats': stats, 'requesterPeakRss': peak_rss(),
              'pass': {'groups': all(g['relative'] <= 1e-5 for g in groups.values()), 'layer': float(err.max()) <= 1e-3}}
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ['groups', 'layerMaxAbs', 'layerScale', 'pass', 'requesterPeakRss']}, indent=1))


def run_prompts(args, model, config, manifest, cluster=None, forced=None):
    tokenizer, rows = tokenize_prompts(args.model, read_json(args.prompts))
    results = []
    for index, row in enumerate(rows):
        t0 = time.perf_counter()
        tokens, steps, logits = greedy(model, config, manifest, row['ids'], args.max_tokens, cluster,
                                       forced=forced[index]['tokens'] if forced else None)
        results.append({'prompt': row['prompt'], 'promptTokens': len(row['ids']), 'tokens': tokens,
                        'text': tokenizer.decode(tokens, skip_special_tokens=False), 'steps': steps,
                        'wallS': time.perf_counter() - t0,
                        'logitsSha256': sha256_bytes(b''.join(l.tobytes() for l in logits))})
        if args.logits_out:
            np.save(f'{args.logits_out}-{index}.npy', np.stack(logits))
        print(json.dumps({'prompt': index, 'tokens': len(tokens), 'wallS': round(results[-1]['wallS'], 2),
                          'text': results[-1]['text'][:80]}), flush=True)
    return results


def cmd_generate(args):
    torch.set_num_threads(args.threads)
    plan = read_json(args.plan)
    t0 = time.perf_counter()
    cluster = SharedCluster(plan, read_json(args.operators), seeded=not args.explicit)
    model, config, manifest = load_model(args.model, args.source_config, plan, cluster)
    load_s = time.perf_counter() - t0
    forced = read_json(args.forced)['results'] if args.forced else None
    results = run_prompts(args, model, config, manifest, cluster, forced)
    stats = cluster.close('exit' if args.exit_operators else 'close')
    digests = [r['shareBSha256'] for r in cluster.rounds]
    counters = [r['counter'] for r in cluster.rounds]
    freshness = {'rounds': len(digests), 'distinctShareBDigests': len(set(digests)),
                 'countersStrictlyIncreasing': all(b > a for a, b in zip(counters, counters[1:])),
                 'shareASeedsPerStage': 'one 32-byte os.urandom seed per stage pair per session, sent only to share A'}
    report = {'freshness': freshness, 'schema': 'doppler.secret-shared-generation/v1', 'modelId': plan['modelId'], 'fx': plan['fx'], 'fw': plan['fw'],
              'seeded': not args.explicit, 'teacherForced': bool(forced), 'loadS': load_s,
              'stages': [{'index': s['index'], 'layers': s['layers']} for s in plan['stages']],
              'results': results, 'operatorStats': stats, 'requesterPeakRss': peak_rss(),
              'rounds': cluster.rounds if args.keep_rounds else None}
    Path(args.out).write_text(json.dumps(report) + '\n')
    print(json.dumps({'freshness': freshness, 'requesterPeakRss': report['requesterPeakRss']}))


def cmd_run(args):
    """One request under a supervisor: events on stdout, `stop` on stdin. Everything else goes to stderr."""
    import threading
    out = sys.stdout
    sys.stdout = sys.stderr
    emit = lambda event: (out.write(json.dumps(event) + '\n'), out.flush())
    stop = threading.Event()

    def watch():
        for line in sys.stdin:
            if line.strip() == 'stop':
                stop.set()
    threading.Thread(target=watch, daemon=True).start()
    torch.set_num_threads(args.threads)
    plan = read_json(args.plan)
    request = read_json(args.request)
    messages = request['messages']
    assert len(messages) == 1 and messages[0]['role'] == 'user', 'one user message'
    try:
        t0 = time.perf_counter()
        cluster = SharedCluster(plan, read_json(args.operators), seeded=not args.explicit)
        model, config, manifest = load_model(args.model, args.source_config, plan, cluster)
        tokenizer, rows = tokenize_prompts(args.model, [messages[0]['content']])
        ids = rows[0]['ids']
        eos = set(manifest['eos_token_id'] if isinstance(manifest['eos_token_id'], list) else [manifest['eos_token_id']])
        emit({'event': 'ready', 'promptTokens': len(ids), 'loadS': time.perf_counter() - t0,
              'stages': [[op.hello['stage'], op.hello['share'], op.hello['planSha256']] for pair in cluster.pairs for op in pair]})
        generated, text = [], ''

        def on_token(step, token, row, done):
            nonlocal text
            generated.append(token)
            full = tokenizer.decode(generated, skip_special_tokens=True)
            delta, text = full[len(text):], full
            reason = 'eos' if token in eos else ('max-tokens' if done else None)
            emit({'event': 'token', 'step': step, 'tokenId': token, 'delta': delta, 'done': bool(done), 'stopReason': reason,
                  'inputTokens': row['columns'], **{k: row.get(k) for k in ['wallMs', 'cpuMs', 'rounds', 'up', 'down', 'wireMs', 'operatorMs', 'margin']}})
        tokens, steps, _ = greedy(model, config, manifest, ids, request['maxTokens'], cluster, on_token=on_token, should_stop=stop.is_set)
        stopped = bool(steps and steps[-1].get('stoppedAfter'))
        stats = cluster.close()
        digests = [r['shareBSha256'] for r in cluster.rounds]
        counters = [r['counter'] for r in cluster.rounds]
        freshness = {'rounds': len(digests), 'distinctShareBDigests': len(set(digests)),
                     'countersStrictlyIncreasing': all(b > a for a, b in zip(counters, counters[1:]))}
        report = {'schema': 'doppler.secret-shared-run/v1', 'modelId': plan['modelId'], 'fx': plan['fx'], 'fw': plan['fw'],
                  'requesterHolds': model.requester_holds,
                  'promptTokens': len(ids), 'tokens': tokens, 'text': text, 'steps': steps, 'freshness': freshness,
                  'operatorStats': stats, 'requesterPeakRss': peak_rss(), 'wallS': time.perf_counter() - t0}
        Path(args.out).write_text(json.dumps(report) + '\n')
        emit({'event': 'end', 'stopReason': 'requester-stop' if stopped else ('eos' if tokens and tokens[-1] in eos else 'max-tokens'),
              'outputTokens': len(tokens), 'freshness': freshness, 'requesterPeakRss': report['requesterPeakRss']})
    except Exception as error:  # the supervisor records the failure; partial text is not a result
        emit({'event': 'failed', 'error': f'{type(error).__name__}: {error}'[:500]})
        sys.exit(1)


def cmd_control(args):
    torch.set_num_threads(args.threads)
    model, config, manifest = load_model(args.model, args.source_config)
    forced = read_json(args.forced)['results'] if args.forced else None
    results = run_prompts(args, model, config, manifest, forced=forced)
    Path(args.out).write_text(json.dumps({'schema': 'doppler.secret-shared-control/v1', 'teacherForced': bool(forced),
                                          'results': results, 'requesterPeakRss': peak_rss()}) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['plan', 'layer', 'generate', 'control', 'run'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-config')
    parser.add_argument('--plan')
    parser.add_argument('--operators')
    parser.add_argument('--prompts')
    parser.add_argument('--request')
    parser.add_argument('--prompt', default='Explain why the sky is blue in one sentence.')
    parser.add_argument('--forced', help='teacher-force the tokens of this report')
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--cuts', default='12')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--fx', type=int, default=18)
    parser.add_argument('--fw', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--explicit', action='store_true', help='send share 0 explicitly instead of from a PRG seed')
    parser.add_argument('--exit-operators', action='store_true')
    parser.add_argument('--keep-rounds', action='store_true')
    parser.add_argument('--logits-out')
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    a.source_config = a.source_config or str(Path(a.model) / 'source-config.json')
    {'plan': cmd_plan, 'layer': cmd_layer, 'generate': cmd_generate, 'control': cmd_control, 'run': cmd_run}[a.command](a)
