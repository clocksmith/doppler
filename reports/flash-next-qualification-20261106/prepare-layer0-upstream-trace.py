"""Observe pinned CPU layer-0 HC/linear stages, stopping before its MoE.

Usage: python3 -P prepare-layer0-upstream-trace.py <producer> <source> <out> <artifact> <reference-data>
       [<prefix-dir> <target-layer>] [<positions-json>]
"""
import hashlib
import json
import sys
from pathlib import Path

assert len(sys.argv) in (6, 7, 8, 9)
producer, source, output, artifact, data = map(Path, sys.argv[1:6])
prefix = Path(sys.argv[6]) if len(sys.argv) >= 8 else None
target_layer = int(sys.argv[7]) if prefix else 0
positions = json.loads(sys.argv[-1]) if len(sys.argv) in (7, 9) else None
assert positions is None or (positions and all(type(p) is int and 0 <= p < 2304 for p in positions))
assert not output.exists()
original = producer.read_bytes()
text = original.decode()
anchor = 'ROOT = Path(__file__).resolve().parent'
assert text.count(anchor) == 1
text = text.replace(anchor, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
if prefix:
    parent_bytes = (prefix / 'receipt.json').read_bytes()
    parent = json.loads(parent_bytes)
    checkpoint = prefix / f'layer-{target_layer - 1}.npy'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == parent['files'][checkpoint.name]
    assert hashlib.sha256(original).hexdigest() == parent['producerSha256']
    assert hashlib.sha256((artifact / 'manifest.json').read_bytes()).hexdigest() == parent['manifestSha256']
    anchor = 'timings = []'
    assert text.count(anchor) == 1
    text = text.replace(anchor, "states['long'] = torch.from_numpy(np.load(" + repr(str(checkpoint.resolve())) + ", allow_pickle=False)).to(DTYPE)\n" + anchor)
    anchor = 'for layer_idx in range(cfg.num_hidden_layers):'
    assert text.count(anchor) == 1
    text = text.replace(anchor, f'for layer_idx in range({target_layer}, {target_layer + 1}):')
anchor = "        hooks = [layer.mlp.gate.register_forward_hook(router_hook)]"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '''
        TRACE_TENSORS = []
        TRACE_STOP = RuntimeError('trace: layer-0 upstream captured')
        def save_trace(role, stage, value, full=False):
            values = value.detach().float().reshape(2304, -1).contiguous().numpy()
            if TRACE_POSITIONS is not None and not full: values = values[TRACE_POSITIONS]
            target = OUT / f'{role}.{stage}.npy'
            np.save(target, values, allow_pickle=False)
            TRACE_TENSORS.append({'role': role, 'stage': stage, 'shape': list(values.shape), 'positions': None if full else TRACE_POSITIONS,
                                  'file': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        def before_hc(role):
            def hook(module, inputs):
                save_trace(role, 'input', inputs[0])
                if role == 'mlp_hc': save_trace('attn_hc', 'injected', inputs[0])
            return hook
        def output_hc(role):
            def hook(module, inputs, result):
                save_trace(role, 'branchInput', result[0])
                save_trace(role, 'gate', result[2])
                if role == 'attn_hc':
                    save_trace('attention', 'projectionInput', result[0], full=True)
                if role == 'mlp_hc': raise TRACE_STOP
            return hook
        def output_hook(role, stage):
            def hook(module, inputs, result):
                save_trace(role, stage, result)
                if stage == 'up': save_trace(role, 'mix', torch.sigmoid(result))
            return hook
        def low_rank(role):
            def hook(module, inputs): save_trace(role, 'lowRank', inputs[0])
            return hook
        for role, module in [('attn_hc', layer.attn_hyper_connection), ('mlp_hc', layer.mlp_hyper_connection)]:
            hooks.append(module.register_forward_pre_hook(before_hc(role)))
            hooks.append(module.register_forward_hook(output_hc(role)))
            for name, stage in [('hc_norm', 'norm'), ('input_mix_weight_down', 'down'),
                                ('input_mix_weight_up', 'up'), ('block_inject_weight', 'logits')]:
                hooks.append(getattr(module, name).register_forward_hook(output_hook(role, stage)))
            hooks.append(module.input_mix_weight_up.register_forward_pre_hook(low_rank(role)))
        if hasattr(layer, 'linear_attn'):
            def linear_output(module, inputs, result):
                save_trace('attn_hc', 'branchOutput', result)
                save_trace('linear', 'branchOutput', result, full=True)
            hooks.append(layer.linear_attn.register_forward_hook(linear_output))
            for name, stage in [('in_proj_qkv', 'attn.qkv_proj'), ('in_proj_z', 'attn.linear_z_proj'),
                                ('in_proj_a', 'attn.linear_a_proj'), ('in_proj_b', 'attn.linear_b_proj'),
                                ('norm', 'attn.linear_core_out')]:
                hooks.append(getattr(layer.linear_attn, name).register_forward_hook(output_hook('linear', stage)))
        else:
            assert KV_F16
            trace_original_core = upstream_attention
            def observe_core(module, query, key, value, *args, **kwargs):
                save_trace('attention', 'query', query.transpose(1, 2), full=True)
                save_trace('attention', 'cachedKey', key.transpose(1, 2), full=True)
                save_trace('attention', 'cachedValue', value.transpose(1, 2), full=True)
                mask = args[0] if args else kwargs['attention_mask']
                save_trace('attention', 'mask', mask)
                result = trace_original_core(module, query, key, value, *args, **kwargs)
                save_trace('attention', 'attn.core_out', result[0])
                return result
            upstream_attention = observe_core
            def attention_output(module, inputs, result):
                save_trace('attn_hc', 'branchOutput', result[0])
            def output_projection_input(module, inputs):
                save_trace('attention', 'gatedCore', inputs[0])
            hooks.append(layer.self_attn.register_forward_hook(attention_output))
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(output_projection_input))
            def value_projection(module, inputs, result):
                save_trace('attention', 'attn.v_proj', result, full=True)
            hooks.append(layer.self_attn.v_proj.register_forward_hook(value_projection))
''')
sys.argv = [str(producer), str(source), str(output), '--dtype', 'float32',
            '--artifact', str(artifact), '--sequences', 'long', '--kv-dtype', 'f16']
environment = {'__name__': '__main__', '__file__': str(producer), 'TRACE_POSITIONS': positions}
try:
    exec(compile(text, str(producer) + ':layer0-upstream-observation', 'exec'), environment)
    raise AssertionError('trace must stop before MoE')
except Exception as error:
    if error is not environment.get('TRACE_STOP'):
        raise
sha = lambda b: hashlib.sha256(b).hexdigest()
report = dict(schema='doppler.flash-next-cpu-upstream-trace/v1', diagnosticCompleted=True,
              scope=f'Pinned CPU upstream stage observations, intentionally stopped before layer-{target_layer} MoE; no GPU values or model qualification',
              normalModelQualified=False, layer=target_layer,
              hyperConnectionGeometry=json.loads((artifact / 'manifest.json').read_bytes())['inference']['hyperConnections'],
              producerSha256=sha(original), instrumentedProducerSha256=sha(text.encode()),
              instrumentSha256=sha(Path(__file__).read_bytes()),
              manifestSha256=sha(environment['ARTIFACT_MANIFEST_BYTES']),
              inputsSha256=sha(environment['INPUTS_PATH'].read_bytes()), tensors=environment['TRACE_TENSORS'])
if prefix:
    assert report['inputsSha256'] == parent['inputsSha256']
    report['parentReceiptSha256'] = sha(parent_bytes)
    report['checkpointSha256'] = parent['files'][checkpoint.name]
(output / 'trace.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(diagnosticCompleted=True, tensors=len(report['tensors']))))
