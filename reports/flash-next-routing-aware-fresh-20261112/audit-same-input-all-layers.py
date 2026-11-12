#!/usr/bin/env python3
"""Same-input operator checks over all 48 layers, the root mixer and the LM head.

Usage:
  python3 -P audit-same-input-all-layers.py --producer-dir PRODUCER_DIR --source-dir SOURCE_DIR \
    --artifact-dir ARTIFACT_DIR --capture-dir CAPTURE_DIR --inputs INPUTS.json --out OUT.json

Extends reports/flash-next-new-prompts-20261110/audit-same-input-operators.py (Layers 0-24 at
one position of a 113-token prefix) to a complete unforced prefill captured by
tests/integration/flash-next-same-input-capture-physical.test.js. Every check feeds the
GPU's own captured input to the pinned upstream CPU module (independently dequantized
stored weights, F32) and scores the GPU's output against that CPU output with the unchanged
doppler.boundary-tolerance/quantized-v1 rule |gpu - cpu| <= 0.05 + 0.05 * |cpu|.

Families (positions: the capture's declared positions; sequence mixers receive the
complete captured input and are scored at those positions):
  token-embedding  Layer-0 stream input vs embed_tokens rows repeated over the 4 streams
  hyper-connection attn/mlp gated residual: norm, branch input and injection, per layer
  gated-delta      linear-attention layers, source chunked rule (the producer's path)
  full-attention   QSA-indexed attention layers
  routed-moe       routed experts with the CPU's own top-k on the GPU input, vs the GPU
                   routed output before the shared-expert add
  shared-expert    GPU routed output + CPU gated shared expert on the GPU input, vs the GPU
                   MoE block output
  final-norm       root hyper-connection mixer (Flash-Next has no final RMSNorm; the mixer
                   is applied in its place before the LM head)
  lm-head          LM head on the GPU's last mixed row vs the GPU's last-position logits
CPU only; never feeds a tensor back into a model.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ABSOLUTE, RELATIVE = 0.05, 0.05
POLICY_ID = 'doppler.boundary-tolerance/quantized-v1'
OLD_AUDIT = Path(__file__).resolve().parents[1] / 'flash-next-new-prompts-20261110' / 'audit-same-input-operators.py'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(gpu, cpu):
    gpu = np.asarray(gpu, dtype=np.float32).astype(np.float64)
    cpu = np.asarray(cpu, dtype=np.float32).astype(np.float64)
    assert gpu.shape == cpu.shape, (gpu.shape, cpu.shape)
    assert np.isfinite(gpu).all() and np.isfinite(cpu).all(), 'non-finite values'
    error = np.abs(gpu - cpu)
    limit = ABSOLUTE + RELATIVE * np.abs(cpu)
    denom = float(np.linalg.norm(cpu.ravel()))
    return {'count': int(error.size), 'outsidePolicy': int(np.sum(error > limit)), 'maxAbs': float(error.max()),
            'relL2': float(np.linalg.norm((gpu - cpu).ravel()) / denom) if denom > 0 else 0.0,
            'maxRatioToLimit': float(np.max(error / limit))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['--producer-dir', '--source-dir', '--artifact-dir', '--capture-dir', '--inputs', '--out']:
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'Refusing to overwrite {args.out}'
    started = time.time()
    spec = importlib.util.spec_from_file_location('audit_same_input_operators', OLD_AUDIT)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    torch.set_num_threads(int(__import__('os').environ.get('AUDIT_THREADS', '8')))
    assert torch.get_float32_matmul_precision() == 'highest'
    ns, cfg, manifest_bytes, read_tensor, handles, pool = base.build_upstream_namespace(
        args.producer_dir, args.source_dir, args.artifact_dir)

    artifact_manifest = json.loads(manifest_bytes)

    def read_expert_subset(name, subset):
        """Independently dequantized Q4_K rows of the named experts only ([len(subset), rows, width])."""
        from gguf import GGMLQuantizationType, quants
        t = artifact_manifest['tensors'][name]
        assert t['dtype'] in ('Q4_K', 'Q4_K_M') and t.get('layout', 'row') == 'row', name
        spans = t.get('spans') or [{'shardIndex': t['shard'], 'offset': t['offset'], 'size': t['size']}]
        raw = bytearray()
        for span in spans:
            with open(args.artifact_dir / artifact_manifest['shards'][span['shardIndex']]['filename'], 'rb') as f:
                f.seek(span['offset'])
                raw += f.read(span['size'])
        assert len(raw) == t['size'], name
        num_e, rows, width = t['shape']
        row_bytes = ((width + 255) // 256) * 144
        encoded = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(num_e, rows, row_bytes)[subset]
        decoded = quants.dequantize(encoded.reshape(-1, row_bytes), GGMLQuantizationType.Q4_K)[:, :width]
        return torch.from_numpy(np.ascontiguousarray(decoded).reshape(len(subset), rows, width))

    obs_path = args.capture_dir / 'observation.json'
    obs = json.loads(obs_path.read_text())
    assert obs['diagnosticCompleted'] is True and obs['gpuErrors'] == [] and obs['cleanupFailures'] == []
    assert obs['manifestSha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert obs['inputsSha256'] == sha256_file(args.inputs)
    tokens = next(s for s in json.loads(args.inputs.read_text())['sequences'] if s['id'] == obs['sequenceId'])['tokens']
    T = len(tokens)
    assert T == obs['tokens']
    P = obs['positions']
    ids = torch.tensor([tokens], dtype=torch.long)

    def load(entry, dtype=np.float32):
        path = args.capture_dir / entry['file']
        assert sha256_file(path) == entry['sha256'], f"capture hash mismatch: {entry['file']}"
        return np.fromfile(path, dtype=dtype).reshape(entry['shape'])

    tensors = {(row['layer'], row['role'], row['stage']): row for row in obs['tensors']}
    routing = {row['layer']: row for row in obs['moeRouting']}
    routed = {row['layer']: row for row in obs['routedOutputs']}
    gpu = lambda layer, role, stage: load(tensors[(layer, role, stage)])

    rotary = ns['Qwen4ExpTextRotaryEmbedding'](cfg)
    probe = torch.zeros(1, T, cfg.hidden_size)
    position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)[1:]
    pos_emb = rotary(probe, position_ids)
    causal_mask = torch.full((T, T), torch.finfo(torch.float32).min, dtype=torch.float32).triu(1)[None, None]
    del probe

    checks = {}

    def add(family, layer, metrics):
        row = checks.setdefault(family, {'family': family, 'caseId': obs['sequenceId'], 'layers': [], 'count': 0,
                                          'outsidePolicy': 0, 'maxAbs': 0.0, 'maxRatioToLimit': 0.0})
        if layer is not None and layer not in row['layers']:
            row['layers'].append(layer)
        row['count'] += metrics['count']
        row['outsidePolicy'] += metrics['outsidePolicy']
        row['maxAbs'] = max(row['maxAbs'], metrics['maxAbs'])
        row['maxRatioToLimit'] = max(row['maxRatioToLimit'], metrics['maxRatioToLimit'])

    def eval_hc(hc_mod, u_np, branch_out_np):
        u = torch.from_numpy(u_np).unsqueeze(0)
        u_norm = hc_mod.hc_norm(u)
        mixed, _, inj_w = hc_mod(u)
        injected = u + (torch.from_numpy(branch_out_np).unsqueeze(0).unsqueeze(-2) * inj_w.unsqueeze(-1)).flatten(-2)
        return u_norm[0].numpy(), mixed[0].numpy(), injected[0].numpy()

    layers = []
    with torch.no_grad():
        embed = read_tensor('model.language_model.embed_tokens.weight')
        cpu_entry = embed[ids[0, P]].to(torch.float32).repeat(1, cfg.hc_count).numpy()
        del embed
        metrics = score(gpu(0, 'attn_hc', 'input'), cpu_entry)
        add('token-embedding', 0, metrics)
        model_rows = {'tokenEmbedding': metrics}

        for l in range(cfg.num_hidden_layers):
            lt0 = time.time()
            layer = ns['Qwen4ExpTextDecoderLayer'](cfg, l).eval()
            prefix = f'model.language_model.layers.{l}.'
            state = {}
            for key, value in layer.state_dict().items():
                if 'ngram_embedding.weight' in key or key.startswith('mlp.experts.'):
                    continue
                tensor = read_tensor(prefix + key)
                assert tuple(tensor.shape) == tuple(value.shape), key
                if value.dtype == torch.int64:
                    assert torch.equal(tensor, value), key
                state[key] = tensor
            layer.load_state_dict(state, strict=False, assign=True)
            layer.to(torch.float32)
            rec = {'layer': l, 'layerType': layer.layer_type}

            for role, module in [('attn_hc', layer.attn_hyper_connection), ('mlp_hc', layer.mlp_hyper_connection)]:
                norm, mixed, injected = eval_hc(module, gpu(l, role, 'input'), gpu(l, role, 'branchOutput'))
                rec[role] = {'norm': score(gpu(l, role, 'norm'), norm),
                             'branchInput': score(gpu(l, role, 'branchInput'), mixed),
                             'injected': score(gpu(l, role, 'injected'), injected)}
                for metrics in rec[role].values():
                    add('hyper-connection', l, metrics)

            full_in = torch.from_numpy(gpu(l, 'attn_hc', 'branchInputAllPositions')).unsqueeze(0)
            if layer.layer_type == 'linear_attention':
                cpu_out = layer.linear_attn(full_in, cache_params=None, attention_mask=None)[0].numpy()
                family = 'gated-delta'
            else:
                cpu_out = layer.self_attn(full_in, pos_emb, attention_mask=causal_mask, past_key_values=None)[0][0].numpy()
                family = 'full-attention'
            rec['sequenceMixer'] = {'family': family, **score(gpu(l, 'attn_hc', 'branchOutput'), cpu_out[P])}
            add(family, l, rec['sequenceMixer'])
            del full_in, cpu_out

            x = torch.from_numpy(gpu(l, 'mlp_hc', 'branchInput'))
            cpu_logits, cpu_scores, cpu_idx = layer.mlp.gate(x)
            shape = routing[l]['shape']
            gpu_idx = load({**routing[l]['indices'], 'shape': shape}, np.uint32).astype(np.int64)[P]
            gpu_w = load({**routing[l]['weights'], 'shape': shape})[P]
            set_diff = [int(P[i]) for i in range(len(P)) if set(cpu_idx[i].tolist()) != set(gpu_idx[i].tolist())]
            active = sorted(set(cpu_idx.flatten().tolist()) | set(gpu_idx.flatten().tolist()))
            # Only the experts the checked positions select are decoded; indices are remapped
            # to that subset, which leaves every selected expert's arithmetic unchanged.
            sub_cfg = type(cfg)(**{**vars(cfg), 'num_experts': len(active)})
            experts = ns['Qwen4ExpTextExperts'](sub_cfg).eval()
            gate_up = read_expert_subset(f'{prefix}mlp.experts.gate_up_proj', active)
            down = read_expert_subset(f'{prefix}mlp.experts.down_proj', active)
            experts.load_state_dict({'gate_up_proj': gate_up, 'down_proj': down}, assign=True)
            local = {expert: i for i, expert in enumerate(active)}
            remap = lambda idx: torch.tensor([[local[int(e)] for e in row] for row in idx], dtype=torch.long)
            cpu_routed = experts(x, remap(cpu_idx), cpu_scores).numpy()
            gpu_routed = load(routed[l])
            rec['routedMoe'] = score(gpu_routed, cpu_routed)
            rec['routedMoe']['cpuVsGpuSelectionDifferingPositions'] = set_diff
            if set_diff:
                # Diagnostic only: the same experts and weights the GPU executed.
                matched = experts(x, remap(gpu_idx), torch.from_numpy(gpu_w)).numpy()
                rec['routedMoe']['withGpuExecutedSelection'] = score(gpu_routed, matched)
            add('routed-moe', l, rec['routedMoe'])
            shared = (torch.sigmoid(layer.mlp.shared_expert_gate(x)) * layer.mlp.shared_expert(x)).numpy()
            rec['sharedExpert'] = score(gpu(l, 'mlp_hc', 'branchOutput'), gpu_routed + shared)
            add('shared-expert', l, rec['sharedExpert'])
            rec['seconds'] = round(time.time() - lt0, 1)
            layers.append(rec)
            print(json.dumps({'layer': l, 'type': layer.layer_type, 'sec': rec['seconds'],
                              'outside': sum(v['outsidePolicy'] for v in checks.values()), 'routingDiffs': len(set_diff)}),
                  flush=True)
            del layer, gate_up, down

        mixer = ns['Qwen4ExpTextGatedResidual'](cfg, use_combine=False).eval()
        mixer_state = {key: read_tensor('model.language_model.hyper_connection_mixer.' + key)
                       for key in mixer.state_dict()}
        mixer.load_state_dict(mixer_state, strict=True, assign=True)
        mixer.to(torch.float32)
        u = torch.from_numpy(gpu('final', 'hc_mixer', 'input')).unsqueeze(0)
        model_rows['finalNormRootMixer'] = {'norm': score(gpu('final', 'hc_mixer', 'norm'), mixer.hc_norm(u)[0].numpy()),
                                            'mixed': score(gpu('final', 'hc_mixer', 'branchInput'), mixer(u)[0].numpy())}
        for metrics in model_rows['finalNormRootMixer'].values():
            add('final-norm', None, metrics)
        head = read_tensor('lm_head.weight').float()
        last_mixed = torch.from_numpy(gpu('final', 'hc_mixer', 'branchInput')[-1])
        cpu_logits_last = (last_mixed @ head.T).numpy()
        gpu_logits = load({'file': obs['prefillLogits']['file'], 'sha256': obs['prefillLogits']['sha256'],
                           'shape': [cpu_logits_last.shape[0]]})
        model_rows['lmHead'] = score(gpu_logits, cpu_logits_last)
        model_rows['lmHead']['argmaxAgrees'] = int(gpu_logits.argmax()) == int(cpu_logits_last.argmax())
        add('lm-head', None, model_rows['lmHead'])

    pool.shutdown(wait=True)
    for h in handles.values():
        h.close()
    for row in checks.values():
        row['layers'].sort()
    out = {
        'schema': 'doppler.routing-aware-same-input-operators/v1',
        'scope': 'Same-input operator checks on the unforced GPU capture of one pinned long case: every layer, root mixer '
                 'and LM head under unchanged quantized-v1; diagnostic evidence, not model qualification',
        'policyId': POLICY_ID, 'tolerance': {'absolute': ABSOLUTE, 'relative': RELATIVE},
        'caseId': obs['sequenceId'], 'tokens': T, 'positions': P,
        'captureObservationSha256': sha256_file(obs_path), 'inputsSha256': sha256_file(args.inputs),
        'artifactManifestSha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'auditScriptSha256': sha256_file(__file__), 'baseAuditSha256': sha256_file(OLD_AUDIT),
        'torchVersion': torch.__version__, 'torchThreads': torch.get_num_threads(),
        'seconds': round(time.time() - started, 1),
        'checks': list(checks.values()), 'model': model_rows, 'layers': layers,
    }
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'written': str(args.out), 'checks': [(c['family'], c['outsidePolicy'], c['count']) for c in checks.values()]}))


if __name__ == '__main__':
    main()
