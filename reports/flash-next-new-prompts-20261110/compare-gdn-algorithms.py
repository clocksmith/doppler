"""CPU-only chunked/recurrent source controls on a retained GPU projection set.

Usage: compare-gdn-algorithms.py GPU_TRACE CPU_TRACE PRODUCER SOURCE ARTIFACT
       DATA VERIFIED_TENSOR_READER NEW_OUTPUT.json [--export-fixture NEW_DIR]
No tensors feed back into a model. GPU comparisons also include CPU/GPU
convolution, decay and norm differences: exact GPU post-convolution inputs
have not been captured by this control.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

arguments = sys.argv[1:]
fixture_path = None
if '--export-fixture' in arguments:
    index = arguments.index('--export-fixture')
    fixture_path = Path(arguments[index + 1])
    assert not fixture_path.exists()
    del arguments[index:index + 2]
gpu_path, cpu_path, producer, source, artifact, data, reader_path, output = map(Path, arguments)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
gpu_raw, cpu_raw = gpu_path.read_bytes(), cpu_path.read_bytes()
gpu, cpu = json.loads(gpu_raw), json.loads(cpu_raw)
assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
assert cpu['diagnosticCompleted'] and cpu['sourceRouterInputReplayBitExact']
assert gpu['manifestSha256'] == cpu['manifestSha256'] == sha((artifact / 'manifest.json').read_bytes())
assert gpu['inputsSha256'] == cpu['inputsSha256']
assert gpu['sequenceId'] == cpu['sequenceId']
assert not output.exists() and not output.with_suffix('.context').exists()
producer_bytes = producer.read_bytes()
assert sha(producer_bytes) == cpu['producerSha256']
text = producer_bytes.decode()
anchor = 'ROOT = Path(__file__).resolve().parent'
assert text.count(anchor) == 1
text = text.replace(anchor, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
marker = '# ---- Inputs ----------------------------------------------------------------------------------'
assert text.count(marker) == 1
text = text.split(marker)[0]
sys.argv = [str(producer), str(source), str(output.with_suffix('.context')),
            '--dtype', 'float32', '--artifact', str(artifact)]
environment = {'__name__': '__main__', '__file__': str(producer)}
exec(compile(text, str(producer) + ':definitions-only', 'exec'), environment)
assert not environment['KV_F16'] and environment['DTYPE'] == torch.float32
namespace = environment['namespace']
spec = importlib.util.spec_from_file_location('verified_tensor_io', reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
layer = cpu['layer']
assert environment['cfg'].layer_types[layer] == 'linear_attention'
module = namespace['Qwen4ExpTextGatedDeltaNet'](environment['cfg'], layer).float().cpu().eval()
environment['load'](module, f'model.language_model.layers.{layer}.linear_attn.')

def gpu_tensor(stage):
    record = next(r for r in gpu['tensors'] if r['layer'] == layer and r['role'] == 'linear' and r['stage'] == stage)
    assert record.get('positions') is None
    return torch.from_numpy(reader.read_tensor(record, gpu_path.parent).copy())

def source_tensor(role, stage):
    record = next(r for r in cpu['tensors'] if r['role'] == role and r['stage'] == stage)
    assert record.get('positions') is None
    return torch.from_numpy(reader.read_tensor(record, cpu_path.parent).copy())

input_values = gpu_tensor('attn.post_input_norm')
tokens = input_values.shape[0]
actual_core = gpu_tensor('attn.linear_core_out').reshape(tokens, -1)
source_input = source_tensor('attention', 'projectionInput')
source_core_record = next(r for r in cpu['tensors'] if r['role'] == 'linear' and r['stage'] == 'attn.linear_core_out')
source_core = torch.from_numpy(reader.read_tensor(source_core_record, cpu_path.parent).copy())
source_positions = source_core_record.get('positions')
if source_positions is not None:
    assert len(set(source_positions)) == len(source_positions) and all(0 <= p < tokens for p in source_positions)
source_core = source_core.reshape(len(source_positions) if source_positions is not None else tokens, -1)
mapping = dict(qkv=('in_proj_qkv', 'attn.qkv_proj'), z=('in_proj_z', 'attn.linear_z_proj'),
               a=('in_proj_a', 'attn.linear_a_proj'), b=('in_proj_b', 'attn.linear_b_proj'))
projections = {key: gpu_tensor(stage) for key, (_, stage) in mapping.items()}

def bits_equal(a, b):
    return np.array_equal(a.detach().numpy().view('u4'), b.detach().numpy().view('u4'))

def metrics(actual, reference):
    assert actual.shape == reference.shape and torch.isfinite(actual).all() and torch.isfinite(reference).all()
    a, b = actual.detach().double().numpy(), reference.detach().double().numpy()
    difference = a - b
    reference_rms = float(np.sqrt(np.mean(b * b)))
    difference_rms = float(np.sqrt(np.mean(difference * difference)))
    return dict(values=int(actual.numel()), maxAbsoluteError=float(np.max(np.abs(difference))),
                referenceRms=reference_rms, differenceRms=difference_rms,
                relativeRms=difference_rms / reference_rms if reference_rms else None,
                differingF32Values=int(np.count_nonzero(actual.detach().numpy().view('u4') != reference.detach().numpy().view('u4'))))

observed = {}
def norm_observer(_module, _inputs, result):
    observed['core'] = result.detach().clone().reshape(tokens, -1)

norm_hook = module.norm.register_forward_hook(norm_observer)
with torch.no_grad():
    module(source_input.unsqueeze(0))
source_replay = observed['core'] if source_positions is None else observed['core'][source_positions]
assert bits_equal(source_replay, source_core), 'Source geometry must reproduce every retained source core value'
original_chunk = namespace['torch_chunk_gated_delta_rule']
original_recurrent = namespace['torch_recurrent_gated_delta_rule']
captured = {}
def observe_chunk(query, key, value, **kwargs):
    assert not captured, 'Expected one cache-free GDN call'
    captured.update(query=query.detach().clone(), key=key.detach().clone(), value=value.detach().clone(),
                    g=kwargs['g'].detach().clone(), beta=kwargs['beta'].detach().clone())
    assert kwargs['initial_state'] is None and not kwargs['output_final_state'] and kwargs['use_qk_l2norm_in_kernel']
    result = original_chunk(query, key, value, **kwargs)
    captured['rawChunk'] = result[0].detach().clone()
    return result
namespace['torch_chunk_gated_delta_rule'] = observe_chunk
handles = []
for key, (field, _) in mapping.items():
    def replace(_module, _inputs, result, key=key):
        return projections[key].reshape(result.shape)
    handles.append(getattr(module, field).register_forward_hook(replace))
try:
    with torch.no_grad():
        module(input_values.unsqueeze(0))
    chunk_core = observed['core'].clone()
finally:
    for handle in handles:
        handle.remove()
    norm_hook.remove()
    namespace['torch_chunk_gated_delta_rule'] = original_chunk

query, key, value = [captured[name] for name in ['query', 'key', 'value']]
kwargs = dict(g=captured['g'], beta=captured['beta'], initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True)
with torch.no_grad():
    chunk_repeat, _ = original_chunk(query, key, value, **kwargs)
    assert bits_equal(chunk_repeat, captured['rawChunk'])
    recurrent, _ = original_recurrent(query, key, value, **kwargs)
    z = projections['z'].reshape(-1, module.head_v_dim)
    recurrent_core = module.norm(recurrent.reshape(-1, module.head_v_dim), z).reshape(tokens, -1)
    # Sanity control: zero beta with a zero initial state cannot write memory.
    zero_beta, _ = original_recurrent(query, key, value, **{**kwargs, 'beta': torch.zeros_like(captured['beta'])})
    assert torch.count_nonzero(zero_beta) == 0 and torch.count_nonzero(chunk_repeat) > 0
    q = query.transpose(1, 2).float().contiguous()
    inverse = torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    separate_q = namespace['l2norm'](q, dim=-1, eps=1e-6) * (q.shape[-1] ** -0.5)
    combined_q = q * (inverse * (q.shape[-1] ** -0.5))

report = dict(schema='doppler.gdn-source-algorithm-control/v1',
    scope='CPU source chunked vs recurrent on identical source-convolved captured GPU projections; GPU comparison also includes convolution/decay/norm differences',
    layer=layer, tokens=tokens, sequenceId=gpu['sequenceId'], newGpuInvocations=0,
    modelQualified=False, acceptanceChanged=False, runtimeChanged=False,
    producerSha256=sha(producer_bytes), definitionsOnlySha256=sha(text.encode()),
    scriptSha256=sha(Path(__file__).read_bytes()), readerSha256=sha(reader_path.read_bytes()),
    gpuTraceSha256=sha(gpu_raw), sourceTraceSha256=sha(cpu_raw),
    manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
    torchVersion=torch.__version__, torchThreads=torch.get_num_threads(),
    sourceGeometryReplayBitExact=True, sourceGeometryCheckedValues=int(source_core.numel()),
    sourceGeometryPositions=source_positions, chunkReplayBitExact=True, zeroBetaControlPassed=True,
    commonInputs={name:dict(shape=list(captured[name].shape), sha256=sha(captured[name].numpy().tobytes()))
                  for name in ['query', 'key', 'value', 'g', 'beta']},
    rawRecurrentVsChunked=metrics(recurrent, chunk_repeat),
    postNormRecurrentVsChunked=metrics(recurrent_core, chunk_core),
    gpuVsSourceChunkedOnGpuProjections=metrics(actual_core, chunk_core),
    gpuVsSourceRecurrentOnGpuProjections=metrics(actual_core, recurrent_core),
    sourceSeparateVsCombinedQScaling=metrics(combined_q, separate_q),
    positions=[dict(position=p, recurrentVsChunked=metrics(recurrent_core[p], chunk_core[p]),
                    gpuVsChunked=metrics(actual_core[p], chunk_core[p]), gpuVsRecurrent=metrics(actual_core[p], recurrent_core[p]))
               for p in [112, tokens - 1]])
if fixture_path is not None:
    fixture_path.mkdir()
    def save(name, values):
        raw = values.detach().numpy().astype('<f4').tobytes()
        with (fixture_path / (name + '.f32')).open('xb') as handle:
            handle.write(raw)
        return dict(file=name + '.f32', bytes=len(raw), sha256=sha(raw))
    fixture = dict(scope='Standalone Layer24 GDN on captured projections; source-convolved CPU algorithm controls',
        manifestSha256=gpu['manifestSha256'], inputsSha256=gpu['inputsSha256'],
        gpuTraceSha256=sha(gpu_raw), sourceTraceSha256=sha(cpu_raw),
        layer=layer, numTokens=tokens,
        geometry=dict(convDim=module.conv_dim, convKernelSize=module.conv_kernel_size,
                      numKHeads=module.num_k_heads, numVHeads=module.num_v_heads,
                      headKDim=module.head_k_dim, headVDim=module.head_v_dim,
                      qSize=module.key_dim, kSize=module.key_dim, valueDim=module.value_dim,
                      qRep=module.num_v_heads // module.num_k_heads, normMode='shared',
                      rmsNormEps=environment['cfg'].rms_norm_eps, qkL2NormEps=1e-6),
        normGateActivation=environment['cfg'].output_gate_type or environment['cfg'].hidden_act,
        weights={name:save(name, values) for name, values in [
            ('convWeight', module.conv1d.weight), ('dtBias', module.dt_bias),
            ('aLog', module.A_log), ('normWeight', module.norm.weight)]},
        sourceCore=save('source-chunked-core', chunk_core),
        sourceRecurrentCore=save('source-recurrent-core', recurrent_core),
        projections={name:next(r for r in gpu['tensors'] if r['layer'] == layer and r['role'] == 'linear' and r['stage'] == stage)
                     for name, (_, stage) in {**mapping, 'core':('norm', 'attn.linear_core_out')}.items()})
    with (fixture_path / 'fixture.json').open('x') as handle:
        handle.write(json.dumps(fixture, indent=2) + '\n')
    report['fixtureSha256'] = sha((fixture_path / 'fixture.json').read_bytes())
with output.open('x') as handle:
    handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2), flush=True)
