"""Offline GDN controls on full captured inputs and projections (Layers 20..22).

Usage: python3 -P check-f32-linear-stage-inputs.py GPU_TRACE CPU_TRACE PRODUCER
       SOURCE ARTIFACT DATA PREFIX_CAPTURE NEW_OUTPUT [--export-fixture NEW_DIR]
The producer preamble supplies its unchanged classes and artifact reader. No
full model executes here and no tensors are fed back into GPU execution.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

arguments = sys.argv[1:]
fixture_directory = None
if '--export-fixture' in arguments:
    index = arguments.index('--export-fixture')
    fixture_directory = Path(arguments[index + 1])
    del arguments[index:index + 2]
gpu_path, cpu_path, producer, source, artifact, data, prefix_path, output = map(Path, arguments)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
gpu_bytes, cpu_bytes, prefix_bytes = [p.read_bytes() for p in [gpu_path, cpu_path, prefix_path]]
gpu, cpu, prefix = map(json.loads, [gpu_bytes, cpu_bytes, prefix_bytes])
assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert cpu['diagnosticCompleted'] and cpu['sourceRouterInputReplayBitExact']
assert gpu['manifestSha256'] == cpu['manifestSha256'] == prefix['manifestSha256'] == sha((artifact / 'manifest.json').read_bytes())
assert gpu['inputsSha256'] == cpu['inputsSha256'] == prefix['inputsSha256']
original = producer.read_bytes()
assert sha(original) == cpu['producerSha256']
text = original.decode()
anchor = 'ROOT = Path(__file__).resolve().parent'
assert text.count(anchor) == 1
text = text.replace(anchor, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
marker = '# ---- Forward ---------------------------------------------------------------------------------'
assert text.count(marker) == 1
text = text.split(marker)[0]
context_directory = output.with_suffix('.context')
assert not output.exists() and not context_directory.exists()
sys.argv = [str(producer), str(source), str(context_directory), '--dtype', 'float32',
            '--artifact', str(artifact), '--sequences', 'long']
environment = {'__name__': '__main__', '__file__': str(producer)}
exec(compile(text, str(producer) + ':definitions-only', 'exec'), environment)
assert not environment['KV_F16'] and environment['DTYPE'] == torch.float32
torch.set_num_threads(16)


def gpu_tensor(record):
    raw = Path(record['file']).read_bytes()
    assert sha(raw) == record['sha256']
    value = np.frombuffer(raw, '<f4').reshape(record['shape']).copy()
    assert np.isfinite(value).all()
    return value


def source_tensor(layer, role, stage):
    record = next(r for r in cpu['sourceTensorsAcrossResumedLayers']
                  if r['layer'] == layer and r['role'] == role and r['stage'] == stage)
    file = cpu_path.parent / record['file']
    assert sha(file.read_bytes()) == record['sha256']
    return np.load(file, allow_pickle=False).copy()


def metrics(actual, reference):
    assert actual.shape == reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    delta = actual.astype('f8') - reference.astype('f8')
    rms = float(np.sqrt(np.mean(reference.astype('f8') ** 2)))
    return dict(values=int(actual.size), maxAbsolute=float(np.max(np.abs(delta))),
                differenceRms=float(np.sqrt(np.mean(delta ** 2))), sourceRms=rms,
                relativeRms=float(np.sqrt(np.mean(delta ** 2)) / rms) if rms else None)


bound_rows = []
for record in gpu['tensors']:
    if record['role'] not in ['attn_hc', 'mlp_hc']:
        continue
    original_record = next(r for r in prefix['tensors'] if r['layer'] == record['layer']
                           and r['role'] == record['role'] and r['stage'] == record['stage'])
    assert record['sha256'] == original_record['sha256'], 'New capture changed frozen prefix trajectory'
    bound_rows.append(dict(layer=record['layer'], role=record['role'], stage=record['stage'], bitExact=True))

results = []
mapping = dict(qkv=('in_proj_qkv', 'attn.qkv_proj'), z=('in_proj_z', 'attn.linear_z_proj'),
               a=('in_proj_a', 'attn.linear_a_proj'), b=('in_proj_b', 'attn.linear_b_proj'))
for layer in [20, 21, 22]:
    assert environment['cfg'].layer_types[layer] == 'linear_attention'
    module = environment['namespace']['Qwen4ExpTextGatedDeltaNet'](environment['cfg'], layer).float().cpu().eval()
    environment['load'](module, f'model.language_model.layers.{layer}.linear_attn.')
    branches = {r['stage']: r for r in gpu['linearBranches'] if r['layer'] == layer}
    input_values, output_values = gpu_tensor(branches['branchInput']), gpu_tensor(branches['branchOutput'])
    actual = {}
    for name, (_field, stage) in {**mapping, 'core': ('norm', 'attn.linear_core_out'),
                                 'input': ('', 'attn.post_input_norm')}.items():
        record = next(r for r in gpu['tensors'] if r['layer'] == layer and r['role'] == 'linear' and r['stage'] == stage)
        actual[name] = gpu_tensor(record)
    assert np.array_equal(actual['input'].view('u4'), input_values.view('u4')), 'Projection input boundary mismatch'
    source_input = source_tensor(layer, 'attention', 'projectionInput')
    source_output = source_tensor(layer, 'linear', 'branchOutput')

    def replay_stages(value, replacements):
        observed, handles = {}, []

        def observe(key):
            def hook(_module, _inputs, result):
                if key in replacements:
                    result = torch.from_numpy(replacements[key]).reshape(result.shape)
                observed[key] = result.detach().reshape(input_values.shape[0], -1).numpy().copy()
                return result if key in replacements else None
            return hook
        for key, (field, _stage) in mapping.items():
            handles.append(getattr(module, field).register_forward_hook(observe(key)))
        handles.append(module.norm.register_forward_hook(observe('core')))
        try:
            with torch.no_grad():
                result = module(torch.from_numpy(value).unsqueeze(0)).squeeze(0).numpy().copy()
        finally:
            for handle in handles:
                handle.remove()
        return result, observed

    source_replay, _ = replay_stages(source_input, {})
    assert np.array_equal(source_replay.view('u4'), source_output.view('u4')), 'Full source geometry must replay bit-exactly'
    on_gpu_input, stages = replay_stages(input_values, {})
    _, forced = replay_stages(input_values, {key: actual[key] for key in mapping})
    repeated, _ = replay_stages(input_values, {})
    assert np.array_equal(repeated.view('u4'), on_gpu_input.view('u4')), 'Same-input CPU replay must be bit-stable'
    with torch.no_grad():
        out_on_gpu_core = module.out_proj(torch.from_numpy(actual['core']).unsqueeze(0)).squeeze(0).numpy().copy()
    results.append(dict(layer=layer, sourceGeometryReplayBitExact=True, sourceReplayCheckedValues=int(source_output.size),
                        sameInputReplayBitStable=True, input=metrics(input_values, source_input),
                        total=metrics(output_values, source_output), localOnSameGpuInput=metrics(output_values, on_gpu_input),
                        propagatedInput=metrics(on_gpu_input, source_output),
                        projections=[dict(stage=stage, **metrics(actual[key], stages[key])) for key, (_field, stage) in mapping.items()],
                        coreOnExactGpuProjections=metrics(actual['core'], forced['core']),
                        outputProjectionOnExactGpuCore=metrics(output_values, out_on_gpu_core)))
    if fixture_directory is not None and layer == 22:
        fixture_directory.mkdir()

        def save_weight(name, value):
            raw = value.detach().numpy().astype('<f4').tobytes()
            (fixture_directory / (name + '.f32')).write_bytes(raw)
            return dict(file=name + '.f32', bytes=len(raw), sha256=sha(raw))

        weights = {name: save_weight(name, value) for name, value in [
            ('convWeight', module.conv1d.weight), ('dtBias', module.dt_bias),
            ('aLog', module.A_log), ('normWeight', module.norm.weight)]}
        raw = forced['core'].astype('<f4').tobytes()
        (fixture_directory / 'source-core.f32').write_bytes(raw)
        cfg = environment['cfg']
        fixture = dict(scope='Learned Layer22 core on actual GPU projections; standalone precision candidate only',
                       normalModelQualified=False, layer=22, numTokens=2304,
                       manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
                       gpuTraceSha256=sha(gpu_bytes), cpuTraceSha256=sha(cpu_bytes),
                       geometry=dict(convDim=module.conv_dim, convKernelSize=module.conv_kernel_size,
                                     numKHeads=module.num_k_heads, numVHeads=module.num_v_heads,
                                     headKDim=module.head_k_dim, headVDim=module.head_v_dim,
                                     qSize=module.key_dim, kSize=module.key_dim, valueDim=module.value_dim,
                                     qRep=module.num_v_heads // module.num_k_heads, normMode='shared',
                                     rmsNormEps=cfg.rms_norm_eps, qkL2NormEps=1e-6),
                       normGateActivation='sigmoid', weights=weights,
                       sourceCore=dict(file='source-core.f32', bytes=len(raw), sha256=sha(raw)),
                       projections={name: next(r for r in gpu['tensors'] if r['layer'] == 22
                                               and r['role'] == 'linear' and r['stage'] == stage)
                                    for name, (_field, stage) in {**mapping, 'core': ('norm', 'attn.linear_core_out')}.items()})
        (fixture_directory / 'fixture.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
    print(json.dumps(results[-1]), flush=True)
    del module

report = dict(scope='Full-sequence CPU GDN on exact captured GPU inputs/projections; not model acceptance',
              normalModelQualified=False, newGpuInvocations=0, runtimeChanged=False, acceptanceChanged=False,
              manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'], gpuTraceSha256=sha(gpu_bytes),
              cpuTraceSha256=sha(cpu_bytes), prefixCaptureSha256=sha(prefix_bytes), producerSha256=sha(original),
              definitionsOnlySha256=sha(text.encode()), scriptSha256=sha(Path(__file__).read_bytes()),
              sourceGeometry=[1, 2304, environment['cfg'].hidden_size], torchVersion=torch.__version__,
              torchThreads=torch.get_num_threads(), frozenPrefixBindings=bound_rows, rows=results)
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
